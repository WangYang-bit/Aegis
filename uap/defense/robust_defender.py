import os
import time
import numpy as np

import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils, common_utils
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from uap.utils.data_utils import compute_index_in_scenario
from uap.utils import train_utils
from uap.models.world_model import WorldModel
from uap.datasets.time_series_dataset import TimeSeriesDataset
from uap.config import data_root, uap_root, uap_path, model_root, len_record
from uap.attaker import UniversalAttacker
from uap.utils.box_utils import compute_iou, check_overlap, compute_overlaps
from uap.tools.feature_analys import get_feature_index, get_random_index
from uap.defense.base_defender import BaseDefender


class robust_defender(BaseDefender):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device
        self.detectors = {}
        self.anchor_boxes = {}
        model_name = "pointpillar_V2VAM"
        model_dir = os.path.join(model_root, model_name)
        config_file = os.path.join(model_dir, "config.yaml")
        hypes = yaml_utils.load_yaml(config_file, None)
        hypes["root_dir"] = os.path.join(data_root, "OPV2V/train")
        hypes["validate_dir"] = os.path.join(data_root, "OPV2V/test")
        detector = train_utils.create_model(hypes)
        detector.to(device)
        initial_epoch, detector = train_utils.load_saved_model(model_dir, detector)
        self.detector = detector.eval()
        self.iou_threshold = config["iou_threshold"]
    
    def defense(self, case, detector):
        result = {}
        output_dict = {}
        batch_data = case["data_dict"]
        output_dict["ego"] = self.detector.fusion_decoder(batch_data)
        with torch.no_grad():
            data_dict = {"ego": batch_data}
            pred_box_tensor, pred_score = self.detector.post_processor.post_process(
                data_dict, output_dict
            )
        result.update({"pred_box_tensor": pred_box_tensor, "pred_score": pred_score})
        return result
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
            
        elif fusion_boxes is None or len(fusion_boxes) == 0:
            remove_label = np.array([True] * gt_bboxes.shape[0])
            remove_error = np.zeros(gt_bboxes.shape[0])
            spoof_label = np.zeros(0)
            spoof_error = np.zeros(0)
            spoof_mask = np.array([]).astype(bool)
            
        else:
            iou = compute_overlaps(gt_bboxes, fusion_boxes)
            if iou.size == 0:
                print("iou size is 0")
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
    defender = robust_defender(config=config, device=device)
    sence_id = 640
    
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
    start_time = time.time()
    v2v_data_dict = v2v_dataset.__getitem__(sence_id)
    v2v_batch_data = v2v_dataset.collate_batch_test([v2v_data_dict])
    v2v_batch_data = train_utils.to_device(v2v_batch_data, device)
    
    fusion_anchor_box = torch.from_numpy(
        fusion_detector.post_processor.generate_anchor_box()
    ).to(device)

    cav_content = v2v_batch_data["ego"]
    gt_box_tensor, gt_obj_id = fusion_detector.post_processor.generate_gt_bbx(v2v_batch_data)
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
    output_dict = uap.attack(data_dict, fusion_detector, attack_dict=attack_dict, apply_attack=True)
    data_dict['sparse_features_2d'] = output_dict['adv_feature']
    # obj_bbox_corners = box_utils.boxes_to_corners_3d(obj_bbox.unsqueeze(0), order=fusion_detector.post_processor.params['order'])[0]
    # iou_attack = compute_iou(obj_bbox_corners, output_dict["pred_box_tensor"])
    # if iou_attack is not None and iou_attack.max() > 0.1:
    #     print("attack failed")
    #     continue

    
    sence_case = {
        'sence_id': sence_id,
        'scenario_index': scenario_id,
        'data_dict': data_dict,
        'fusion_anchor_box': fusion_anchor_box,
        'fusion_box': output_dict["pred_box_tensor"],
        'gt_bboxes': gt_box_tensor,
        'attack_bbox_corners': obj_bbox,
        'attack_bbox': obj_bbox,
        }
    
    result = defender.defense(sence_case, detector=fusion_detector)
    defense_results = {
        "spoof_error": [],
        "spoof_label": [],
        "remove_error": [],
        "remove_label": [],
        "success": [],
    }
    defense_results = defender.defense_evaluation_processor(result, sence_case, defense_results)
    end_time = time.time()
    print(f"Time: {end_time - start_time}")
        
