from collections import OrderedDict
import random
import os
import time
import numpy as np

import torch
import torch.nn.functional as F
import opencood.hypes_yaml.yaml_utils as yaml_utils

from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment


from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils, common_utils
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from uap.utils.data_utils import compute_index_in_scenario
from uap.utils import train_utils
from uap.models.world_model import WorldModel
from uap.datasets.time_series_dataset import TimeSeriesDataset
from uap.config import data_root, uap_root, uap_path, model_root, len_record
from uap.attaker import UniversalAttacker
from uap.tools.feature_analys import get_feature_index, get_random_index
from uap.defense.base_defender import BaseDefender
from uap.utils.box_utils import compute_overlaps, compute_iou

# Hungarian Matching
def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))



class robosac_defender(BaseDefender):
    def __init__(self, config, device):
        super(robosac_defender, self).__init__()
        self.device = device
        self.config = config
        self.step_budget = 7
        self.iou_threshold = config["iou_threshold"]
        self.box_matching_thresh = 0.3

    def cal_robosac_steps(self, num_agent, num_consensus, num_attackers):
        # exclude ego agent
        num_agent = num_agent - 1
        eta = num_attackers / num_agent
        # print(f'eta: {eta}')
        # print(f's(num_agent): {num_agent}')
        N = np.ceil(np.log(1 - 0.99) / np.log(1 - np.power(1 - eta, num_consensus))).astype(int)
        return N
    
    def cal_robosac_consensus(self, num_agent, step_budget, num_attackers):
        num_agent = num_agent - 1
        eta = num_attackers / num_agent
        s = np.floor(np.log(1-np.power(1-0.99, 1/step_budget)) / np.log(1-eta)).astype(int)
        return s


    def defense(self, case, detector):
        """
        Perform spatial consistency check on the given case.
        This method checks the consistency of predicted bounding boxes from
        multiple cooperative autonomous vehicles (CAVs) with the fused bounding box.
        Args:
            case (dict): A dictionary containing the following keys:
                - "data_dict" (tensor): The data dictionary containing the features of all agents.
                - "car_position" (tensor): The position of the ego vehicle.
                - "victim_id" (int): The ID of the victim vehicle.
                - "single_cav_output" (dict): A dictionary where each key is a CAV ID
                  and each value is another dictionary containing:
                    - "transformation_matrix" (tensor): Transformation matrix for the CAV.
                    - "pred_box" (tensor): Predicted bounding boxes from the CAV.
                    - "pred_score" (tensor): Prediction scores for the bounding boxes.
                    - "confidence_map" (tensor): Confidence map for the predicted results.
                - "fusion_box" (tensor): The fused bounding box.
        Returns:
            bool: True if the case is ttacked, False otherwise.
        """
         # STEP 1:
        # get original ego agent class prediction of all anchors, without adv pert and fuse, return cls pred of all agents

        # There are attackers among us: 

        # STEP 2:
        # generate adv perturb
        result = {}
        data_dict = case["data_dict"]
        muti_cav_feature = data_dict["spatial_features_2d"]

        ego_idx = 1
        num_agent = len(muti_cav_feature)
        all_agent_list = [i for i in range(num_agent)]
        if ego_idx in all_agent_list:
            all_agent_list.remove(ego_idx)
        else:
            print("Ego agent is not in the list of all agents")
            result.update({"pred_box_tensor": None})
            result.update({"attacker_list": []})
            result.update({"consensus_size": 0})
            return result
        teammate_num = num_agent - 1
        # Randomly samples neighboring agents as attackers
        # NOTE: 

        estimate_attacker_ratio = [i/teammate_num for i in range(0, teammate_num)]
        estimated_attacker_ratio = 1.0
        NMax = []
        for ratio in estimate_attacker_ratio:
            # TODO: set 5 to a variable
            temp_num_attackers = round(teammate_num * (ratio))
            temp_num_consensus = teammate_num - temp_num_attackers
            NMax.append(self.cal_robosac_steps(num_agent, temp_num_consensus, temp_num_attackers))
        
        # Special case when assuming all agents are benign.(i.e. attacker ratio = 1.0)
        # means once if we can't test consensus in 1 try, there's definitely at least 1 attacker.
        NMax[0] = 1
        # print("NMax:", NMax)
        # {5: 1, 4: 9, 3: 19, 2: 27, 1: 21}
        NTry = [0] * len(estimate_attacker_ratio)
        total_sampling_step =0
        # Given Step Budget N and Sampling Set Size s, perform predictions

        step = 0

        ego_result = self.sample_fusion(muti_cav_feature, [ego_idx], data_dict, detector)
        ego_pred_box = ego_result["pred_box_tensor"]

        succ_result = [ego_idx]
        succ_box = ego_pred_box
        succ_probing_consensus_size = 0
        while step < self.step_budget and NTry < NMax:
            for i in range(len(estimate_attacker_ratio)):
                temp_attacker_ratio = estimate_attacker_ratio[i]
                consensus_set_size = round(teammate_num*(1-temp_attacker_ratio))
                if NTry[i] < NMax[i]:
                    # print("Probing {} agents for consensus".format(consensus_set_size))
                    step += 1
                    total_sampling_step += 1
                    # probing_step_tried_by_consensus_set_size[consensus_set_size] += 1
                    # step budget available for probing
                    # try to probe attacker ratio
                    collab_agent_list = random.sample(
                    all_agent_list, k=consensus_set_size)
                    collab_agent_list.append(ego_idx)
                    sample_result = self.sample_fusion(
                        muti_cav_feature, collab_agent_list, data_dict, detector)
                    sample_pred_box = sample_result["pred_box_tensor"]
                    # We use jaccard index to define the difference between two bbox sets
                    jac_index = self.associate_2_detections(
                        detections1=ego_pred_box, detections2=sample_pred_box, iou_threshold=self.iou_threshold)

                    if jac_index < self.box_matching_thresh:
                        # fail to reach consensus
                        # print('No consensus reached when probing {} consensus agents. Current step is {}.'.format(consensus_set_size,step))
                        # print('Attacker(s) is(are) among {}'.format(collab_agent_list))

                        NTry[i] += 1 
                        
                        # if temp_num_attackers == 0:
                        #     # Assumption of no attackers fails
                        #     consensus_tries_is_needed[i] = 0

                        # if NTry[i] == NMax[i]:
                        #     print("Probing of {} agents for consensus has reached its sampling limit {} with assumed attacker ratio {} and consensus set size {}.".format(consensus_set_size, NMax[i], temp_attacker_ratio, consensus_set_size))
                        #     print("From now on we won't try to probe {} agents consensus since it seems unlikely to reach that.".format(consensus_set_size))
                    else:
                        # succeed to reach consensus
                        sus_agent_list = [
                            i for i in all_agent_list if i not in collab_agent_list]
                        # print('Achieved consensus at step {}, with {} agents: {}. Using the result as temporal final output of this frame, and skipping smaller consensus set tries. \n Attacker(s) is(are) among {}, excluded.'.format(
                        #     step, consensus_set_size, collab_agent_list, sus_agent_list))
                        
                        succ_result = sus_agent_list
                        succ_box = sample_pred_box
                        succ_probing_consensus_size = consensus_set_size
                        
                        if temp_attacker_ratio < estimated_attacker_ratio:
                            # print('Larger consensus set ({} agents) probed. We will skip all the smaller consensus set tries. Update attacker ratio estimation to {}'.format(consensus_set_size, temp_attacker_ratio))
                            estimated_attacker_ratio = temp_attacker_ratio
                            
                            for j in range(i, len(estimate_attacker_ratio)):
                                # set all the larger attacker ratio to 0
                                NTry[j] = NMax[j]

                            break 
        result.update({"pred_box_tensor": succ_box})
        result.update({"attacker_list": succ_result})
        result.update({"consensus_size": succ_probing_consensus_size})
        
        return result
                    
    def sample_fusion(self, spatial_feature, collab_agent_list, batch_data, detector):
        output_dict = OrderedDict()
        output_dict = {}
        # Select features from spatial_feature using the indices in collab_agent_list
        spatial_feature_2d = spatial_feature[collab_agent_list]
        # Construct record length as a tensor of ones with length equal to number of selected agents
        record_len = torch.tensor([len(collab_agent_list)], dtype=torch.int32, device=spatial_feature.device)
        # Build data dictionary to feed into the fusion decoder
        data_dict = {
            'spatial_features_2d': spatial_feature_2d,
            'record_len': record_len,
        }
        # Decode fusion result
        output_dict["ego"] = detector.fusion_decoder(data_dict)
        with torch.no_grad():
            data_dict = {"ego": batch_data}
            pred_box_tensor, pred_score = detector.post_processor.post_process(
                data_dict, output_dict
            )
        result = {
            "pred_box_tensor": pred_box_tensor,
            "pred_score": pred_score,
            "rm": output_dict["ego"]["rm"],
            "psm": output_dict["ego"]["psm"]
        }
        return result
    
    def associate_2_detections(self, detections1, detections2, iou_threshold=0.5):
        # Boxes assigned by Hungarian Matching is considered a match, contribute 1 intersect item.
        # This function returns the "IoU" of two bbox sets.

        # if detections2 is empty，directly return 0 associations.
        if detections2 is None or detections1 is None:
            return 0

        iou_matrix = compute_overlaps(detections1, detections2)
        # [[0.73691421 0.         0.         0.        ]
        #  [0.         0.89356082 0.         0.        ]
        #  [0.         0.         0.76781823 0.        ]]
    
        if min(iou_matrix.shape) > 0:
    
            a = (iou_matrix > iou_threshold).astype(np.int32)
            # [[1 0 0 0]
            #  [0 1 0 0]
            #  [0 0 1 0]]
    
            # print(a.sum(1)): [1 1 1]
            # print(a.sum(0)): [1 1 1 0]
    
            # if box with IoU > 0.5 has one-one matching，straight return the result. Or use hungarian matching
            if a.sum(1).max() == 1 and a.sum(0).max() == 1:
    
                matched_indices = np.stack(np.where(a), axis=1)
                # [[0 0]
                #  [1 1]
                #  [2 2]]
            else:
                matched_indices = linear_assignment(-iou_matrix)
        else:
            matched_indices = np.empty(shape=(0, 2))
    
        unmatched_detections1 = []
        for d, det in enumerate(detections1):
            if d not in matched_indices[:, 0]:
                unmatched_detections1.append(d)
    
        unmatched_detections2 = []
        for t, det in enumerate(detections2):
            if t not in matched_indices[:, 1]:
                unmatched_detections2.append(t)
    
        # print(unmatched_detections1) : []
        # print(unmatched_detections2) : [3]
    
        # if matches after hungarian matching has low IoU, also consider them as mis-match.
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections1.append(m[0])
                unmatched_detections2.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
    
        # print(matches): [[0 0] [1 1] [2 2]]
        # print(np.array(unmatched_detections1)): []
        # print(np.array(unmatched_detections2)): [3]
        intersect = len(matches)
        union = len(detections1) + len(detections2) - intersect
        jaccard_index = intersect / union
    
        return jaccard_index

    def caculate_max_iou(self, pred_boxes, fusion_boxes):
        """
        Calculate the maximum IoU score between the predicted bounding boxes and the fused bounding box.
        Args:
            pred_box (Tensor): Predicted bounding boxes from the CAV.
            fusion_box (Tensor): The fused bounding box.
        Returns:
            float: The maximum IoU score.
        """
        pred_iou_list = []
        fusion_iou_list = []
        if (pred_boxes is None or len(pred_boxes) == 0) and (fusion_boxes is None or len(fusion_boxes) == 0):
            return pred_iou_list, fusion_iou_list
        elif pred_boxes is None or len(pred_boxes) == 0:
            fusion_iou_list = [0 for _ in range(len(fusion_boxes))]
            return pred_iou_list, fusion_iou_list
        elif fusion_boxes is None or len(fusion_boxes) == 0:
            pred_iou_list = [0 for _ in range(len(pred_boxes))]
            return pred_iou_list, fusion_iou_list
            

        pred_boxes = common_utils.torch_tensor_to_numpy(pred_boxes)
        fusion_boxes = common_utils.torch_tensor_to_numpy(fusion_boxes)
        
        pred_polygon_list = list(common_utils.convert_format(pred_boxes))
        fusion_polygon_list = list(common_utils.convert_format(fusion_boxes))
        for p in pred_polygon_list:
            ious = common_utils.compute_iou(p, fusion_polygon_list)
            pred_iou_list.append(ious.max())
            
        for f in fusion_polygon_list:
            ious = common_utils.compute_iou(f, pred_polygon_list)
            fusion_iou_list.append(ious.max())
        return pred_iou_list, fusion_iou_list


    def find_unmatched_boxes(self, pred_boxes, fusion_boxes, iou_threshold=0.7):
        """
        Compare predicted boxes and fused boxes by IoU and determine mismatches.
        Boxes in pred_box not matched are 'remove'; boxes in fusion_box not matched are 'spoof'.
        """
        remove_list = []
        spoof_list = []
        pred_iou_list, fusion_iou_list = self.caculate_max_iou(pred_boxes, fusion_boxes)

        for i, iou in enumerate(pred_iou_list):
            if iou < iou_threshold:
                remove_list.append(pred_boxes[i])

        for j, iou in enumerate(fusion_iou_list):
            if iou < iou_threshold:
                spoof_list.append(fusion_boxes[j])

        return {"remove": remove_list, "spoof": spoof_list}
    
    def defense_evaluation_processor(self, result, case, defense_results):
        fusion_boxes = case["fusion_box"]
        gt_bboxes = case["gt_bboxes"]
        attack_bbox = case["attack_bbox_corners"]
        robust_fusion_boxes = result["pred_box_tensor"]
        if robust_fusion_boxes is None:
            return defense_results
        is_success = False 
        if gt_bboxes is None and fusion_boxes is None:
            return defense_results
        elif gt_bboxes is None:
            spoof_label = np.array([True] * fusion_boxes.shape[0])
            remove_label = np.zeros(0)
            remove_error = np.zeros(0)
            remove_mask = np.array([]).astype(bool)
            spoof_error = np.zeros(fusion_boxes.shape[0])
            
        elif fusion_boxes is None:
            remove_label = np.array([True] * gt_bboxes.shape[0])
            remove_error = np.zeros(gt_bboxes.shape[0])
            spoof_label = np.zeros(0)
            spoof_error = np.zeros(0)
            spoof_mask = np.array([]).astype(bool)
            
        else:
            iou = compute_overlaps(gt_bboxes, fusion_boxes)
            spoof_label = np.max(iou, axis=0) <= self.iou_threshold
            remove_label = np.max(iou, axis=1) <= self.iou_threshold
            spoof_error = np.zeros(fusion_boxes.shape[0])     
            remove_error = np.zeros(gt_bboxes.shape[0])
        
        unmatched_boxes = self.find_unmatched_boxes(robust_fusion_boxes, fusion_boxes, iou_threshold=self.iou_threshold)
        spoof_list = unmatched_boxes["spoof"]
        remove_list = unmatched_boxes["remove"]
        
        for s_box in spoof_list:
            iou_val = compute_iou(s_box, fusion_boxes)
            if iou_val.max() > self.iou_threshold:
                idx = np.argmax(iou_val)
                spoof_error[idx] = 1

        for r_box in remove_list:
            iou_val = compute_iou(r_box, gt_bboxes)
            if iou_val.max() > self.iou_threshold:
                idx = np.argmax(iou_val)
                remove_error[idx] = 1

        remove_iou = compute_iou(attack_bbox, remove_list)
        spoof_iou = compute_iou(attack_bbox, spoof_list)
        if remove_iou.max() > self.iou_threshold or spoof_iou.max() > 0.3:
            is_success = True

        defense_results["spoof_error"].append(spoof_error)
        defense_results["spoof_label"].append(spoof_label)
        defense_results["remove_error"].append(remove_error)
        defense_results["remove_label"].append(remove_label)
        defense_results["success"].append(np.array([is_success]).astype(np.int8))
        return defense_results

    def compute_roc(self, value, label, show=False, save=None):
        tpr_data = []
        fpr_data = []
        roc_auc = 0
        best_thres = 0
        best_TPR = 0
        best_FPR = 0
        for thres in np.arange(value.min()-0.02, value.max()+0.02, 0.02).tolist():
            TP = np.sum((value > thres) * (label > 0))
            FP = np.sum((value > thres) * (label <= 0))
            P = np.sum(label > 0)
            N = np.sum(label <= 0)
            PP = TP + FP
            PN = P + N - PP
            TPR = TP / P
            FPR = FP / N
            if TPR * (1 - FPR) > roc_auc:
                roc_auc = TPR * (1 - FPR)
                best_thres = thres
                best_TPR = TPR
                best_FPR = FPR
            tpr_data.append(TPR)
            fpr_data.append(FPR)

        return best_TPR, best_FPR, roc_auc, best_thres, tpr_data, fpr_data

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_file = os.path.join(
        uap_path, "configs/uap.yaml"
    )
    config = yaml_utils.load_yaml(config_file, None)
    defender = robosac_defender(config=config, device=device)
    sence_id = 126
    
    index, scenario_id = compute_index_in_scenario(sence_id, len_record)
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")
    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = os.path.join(data_root, "train")
    hypes["validate_dir"] = os.path.join(data_root, "test")
    v2v_dataset = build_dataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(
        v2v_dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=v2v_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    
    uap = UniversalAttacker(config, device)
    fusion_detector = uap.detectors["pointpillar_V2VAM"]
    single_cav_detector = uap.detectors["pointpillar_single_car_large"]
    start_time = time.time()
    v2v_data_dict = v2v_dataset.__getitem__(sence_id)
    v2v_batch_data = v2v_dataset.collate_batch_test([v2v_data_dict])
    v2v_batch_data = train_utils.to_device(v2v_batch_data, device)
    gt_box_tensor, gt_obj_id = fusion_detector.post_processor.generate_gt_bbx(v2v_batch_data)
    
    fusion_anchor_box = torch.from_numpy(
        fusion_detector.post_processor.generate_anchor_box()
    ).to(device)

    cav_content = v2v_batch_data["ego"]

    # cav_content["lidar_pose"] = np.array(cav_content["lidar_pose"])

    obj_index_dict = get_random_index(cav_content, v2v_dataset, 1)
    obj_id = list(obj_index_dict.keys())[0]
    feature_index, obj_bbox = obj_index_dict[obj_id]
    
    data_dict = fusion_detector.feature_encoder(cav_content)
    data_dict["anchor_box"] = fusion_anchor_box
    
    attack_dict = {"attack_tagget": "remove"}
    attack_dict["obj_idx"] = feature_index
    attack_dict["object_bbox"] = obj_bbox

    # print(attack_dict)
    output_dict = uap.attack(data_dict, fusion_detector, attack_dict=attack_dict, apply_attack=False)
    data_dict['sparse_features_2d'] = output_dict['adv_feature']

    # obj_bbox = box_utils.boxes_to_corners_3d(obj_bbox.unsqueeze(0), order=fusion_detector.post_processor.params['order'])[0]
    # data_dict["anchor_box"] = single_cav_anchor_box
    # single_cav_output = uap.attack_single_car(data_dict, single_cav_detector, attack_dict=attack_dict, apply_attack=False)
    # iou_attack = compute_iou(obj_bbox, output_dict["pred_box_tensor"])
    # if iou_attack is not None and iou_attack.max() > 0.1:
    #     print("attack failed")
    defense_results = {
            "spoof_error": [],
            "spoof_label": [],
            "spoof_location": [],
            "remove_error": [],
            "remove_label": [],
            "remove_location": [],
            "success": [],
            "total_attacker_num": 0
    }

    
    sence_case = {
        'sence_id': sence_id,
        'scenario_index': scenario_id,
        'data_dict': data_dict,
        'fusion_feature': output_dict['fused_feature'],
        "fusion_box": output_dict['pred_box_tensor'],
        'fusion_anchor_box': fusion_anchor_box,
        'fusion_output': output_dict,
        "gt_bboxes": gt_box_tensor,
        "ego_lidar_pose": cav_content["lidar_pose"],
        'single_cav_anchor_box': fusion_anchor_box,
        "attack_bbox": obj_bbox,
        "attack_bbox_corners": obj_bbox,
        }
    result = {"remove": [], "spoof": []}
    result = defender.defense(sence_case, detector=fusion_detector)
    defense_results = defender.defense_evaluation_processor(result, sence_case, defense_results)    
