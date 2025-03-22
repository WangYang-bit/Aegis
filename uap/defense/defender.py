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


def get_car_position(cav_content):
    object_ids = cav_content["object_ids"]
    object_bbx_center = cav_content["object_bbx_center"]
    # If object_bbx_center is a tensor, convert it to a numpy array
    if hasattr(object_bbx_center, "cpu"):
        object_bbx_center = object_bbx_center.cpu().numpy()
    # object_bbx_center has a shape of (1, N, 7); extract the first three elements as [x, y, z] for each of the N objects
    centers = object_bbx_center[0, :, :3]
    
    cav_list = cav_content["cav_list"]
    car_position = {}
    for cav_id in cav_list:
        if cav_id in object_ids:
            index = object_ids.index(cav_id)
            car_position[cav_id] = centers[index]  # [x, y, z]
        else:
            print(f"Warning: cav_id {cav_id} not found in object_ids")
    return car_position




class spatial_temporal_defender(BaseDefender):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.device = device
        self.iou_thres = config["iou_threshold"]
        self.params = config["defense_params"]
        self.confidence_threshold = self.params["confidence_threshold"]
        self.range = self.params["range"]
        # [x , y , z] grid size in x, y, z direction
        self.grid_size = self.params["grid_size"]
        self.consistent_score_threshold = self.params["consistent_score_threshold"]
        
        
        model_args = config["model"]["world_model"]
        self.worldmodel = WorldModel(model_args, device=device)
        self.condition_frame = self.worldmodel.condition_frames
        assert config["train_params"]["init_model_path"] is not None, "model path is None"
        init_epoch, self.worldmodel = train_utils.load_saved_model(
            config["train_params"]["init_model_path"], self.worldmodel
        )
        self.worldmodel.to(device)
        
        dataset_params = config["data"]
        self.time_series_dataset = TimeSeriesDataset(dataset_params, train=False)
    
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
                remove_list.append(i)

        for j, iou in enumerate(fusion_iou_list):
            if iou < iou_threshold:
                spoof_list.append(j)

        return {"remove": remove_list, "spoof": spoof_list}

    def filter_unmatched_boxes(self, unmatched_boxes, confidence_map, threshold):
        """
        Filter out boxes from remove_list and spoof_list based on confidence map values.

        Parameters:
        unmatched_boxes (dict): Dictionary containing 'remove' and 'spoof' lists of boxes.
        confidence_map (numpy.ndarray): Confidence map with dimensions (h, w).
        threshold (float): Threshold value for confidence.

        Returns:
        dict: Filtered unmatched_boxes with boxes having average confidence above the threshold.
        """

        def get_average_confidence(box, confidence_map):
            """
            Calculate the average confidence for a given box based on the confidence map.

            Parameters:
            box (numpy.ndarray): Box with dimensions (8, 3) representing 8 vertices with (x, y, z) coordinates.
            confidence_map (numpy.ndarray): Confidence map with dimensions (h, w).

            Returns:
            float: Average confidence value for the box.
            """
            h, w = confidence_map.shape
            total_confidence = 0
            count = 0
            min_x, min_y = box[:, 0].min(), box[:, 1].min()
            max_x, max_y = box[:, 0].max(), box[:, 1].max()
            # Convert real-world coordinates to confidence map coordinates
            min_map_x = int((min_x - self.range[0]) / self.grid_size[0])
            max_map_x = int((max_x - self.range[0]) / self.grid_size[0])
            min_map_y = int((min_y - self.range[1]) / self.grid_size[1])
            max_map_y = int((max_y - self.range[1]) / self.grid_size[1])
            for map_x in range(min_map_x, max_map_x + 1):
                for map_y in range(min_map_y, max_map_y + 1):
                    if 0 <= map_x < w and 0 <= map_y < h:
                        total_confidence += confidence_map[map_y, map_x]
                        count += 1
            return total_confidence / count if count > 0 else 0

        filtered_remove_list = [
            box
            for box in unmatched_boxes["remove"]
            if get_average_confidence(box, confidence_map) >= threshold
        ]
        filtered_spoof_list = [
            box
            for box in unmatched_boxes["spoof"]
            if get_average_confidence(box, confidence_map) >= threshold
        ]

        return {"remove": filtered_remove_list, "spoof": filtered_spoof_list}

    def filter_boxes_by_range(self, boxes, center):
        """
        Filters out boxes that are outside the specified range around the center.

        Parameters:
        boxes (Tensor): A tensor of shape (N, 8, 3) representing N boxes.
        center (numpy.ndarray): A numpy array of shape (3,) representing the (x, y, z) coordinates of the center.

        Returns:
        Tensor: A tensor containing the boxes 
        """
        if boxes is None or len(boxes) == 0:
            return boxes
        # Convert center and self.range to tensors with the same dtype and device as boxes
        center_tensor = torch.tensor(center, dtype=boxes.dtype, device=boxes.device)
        
        range_tensor = torch.tensor(self.range, dtype=boxes.dtype, device=boxes.device)
        lower_bound = center_tensor - range_tensor
        upper_bound = center_tensor + range_tensor

        # Compute the minimum and maximum coordinates for each box along dimension 1 (8 vertices)
        min_vals = boxes.min(dim=1).values  # shape: (N, 3)
        max_vals = boxes.max(dim=1).values  # shape: (N, 3)

        # Check if each box is completely within the lower and upper bounds
        min_mask = (min_vals >= lower_bound).all(dim=1)
        max_mask = (max_vals <= upper_bound).all(dim=1)

        mask = min_mask & max_mask
        return boxes[mask]
                     
    def spatial_consistent_check(self, case, detector=None):
        """
        Perform spatial consistency check on the given case.
        This method checks the consistency of predicted bounding boxes from
        multiple cooperative autonomous vehicles (CAVs) with the fused bounding box.
        Args:
            case (dict): A dictionary containing the following keys:
                - "single_cav_output" (dict): A dictionary where each key is a CAV ID
                  and each value is another dictionary containing:
                    - "transformation_matrix" (tensor): Transformation matrix for the CAV.
                    - "pred_box" (tensor): Predicted bounding boxes from the CAV.
                    - "pred_score" (tensor): Prediction scores for the bounding boxes.
                    - "confidence_map" (tensor): Confidence map for the predicted results.
                - "fusion_box" (tensor): The fused bounding box.
        Returns:
            list: A list of filtered unmatched bounding boxes that do not match the
                  fused bounding box and have confidence scores above the threshold.
        """

        single_cav_output = case["single_cav_output"]
        fusion_output = case["fusion_output"]
        fusion_box = fusion_output["pred_box_tensor"]
        car_positon = case["car_position"]
        anchor_box = case["single_cav_anchor_box"]
        result = {"remove": [],'remove_score': [], "spoof": [], 'spoof_score': []}
        for cav_id, cav_result in single_cav_output.items():
            # if cav_id == next(iter(single_cav_output)):
            #     continue
            ego_fusion_box = self.filter_boxes_by_range(fusion_box, car_positon[cav_id])
            pred_box = cav_result['pred_box_tensor']
            pred_box = self.filter_boxes_by_range(pred_box, car_positon[cav_id])
            pred_score = cav_result["pred_score"]
            unmatched_boxes = self.find_unmatched_boxes(pred_box, ego_fusion_box, self.iou_thres)

            prob = F.sigmoid(cav_result["psm"].permute(1, 2, 0)).reshape(-1)
            proposals = VoxelPostprocessor.delta_to_boxes3d(
                cav_result["rm"].unsqueeze(0), anchor_box
            )[0]
            proposals = box_utils.boxes_to_corners_3d(proposals, order=detector.post_processor.params['order'])
            
            for remove_index in unmatched_boxes["remove"]:
                remove_box = pred_box[remove_index]
                score = self.consistent_score(remove_box, proposals, prob)
                # if score > self.consistent_score_threshold:
                result["remove"].append(remove_box)
                result["remove_score"].append(score - self.consistent_score_threshold)
            for spoof_index in unmatched_boxes["spoof"]:
                spoof_box = ego_fusion_box[spoof_index]
                score = self.consistent_score(spoof_box, proposals, prob)
                # if score < self.consistent_score_threshold:
                result["spoof"].append(spoof_box)
                result["spoof_score"].append(self.consistent_score_threshold - score)
        return result
    
    def consistent_score(self, pred_box, proposals, prob, order='hwl'):
        """
        Calculate the consistency score between the predicted bounding boxes and the fused bounding box.
        The score is calculated as the average of the IoU scores between the predicted bounding boxes
        and the fused bounding box.
        Args:
            pred_box (np.ndarray): Predicted bounding boxes from the CAV.
            proposals (np.ndarray): The proposals generated by the world model.
            pro (np.ndarray): The probability of the proposals.
        Returns:
            float: The consistency score.
        """
        overlap_mask = check_overlap(pred_box, proposals, order)
        proposals = proposals[overlap_mask]
        prob = prob[overlap_mask]
        ious = compute_iou(pred_box, proposals)
        # pred_box = pred_box.unsqueeze(0)
        # ious_tensor = boxes_iou3d_gpu(pred_box, proposals)
        ious_tensor = torch.from_numpy(ious).to(prob.device)
        box_mask = ious_tensor >= 0.01
        consistent_score = (-ious_tensor[box_mask] * torch.log(1 - prob[box_mask])).sum()
        return consistent_score     

    def temporal_consistent_check(self, case, detector=None):
        """
        Perform temporal consistency check on the given case.
        This method checks the consistency of predicted bounding boxes from
        multiple cooperative autonomous vehicles (CAVs) with the fused bounding box.
        Args:
            case (dict): A dictionary containing the following keys:
                - "single_cav_result" (dict): A dictionary where each key is a CAV ID
                  and each value is another dictionary containing:
                    - "transformation_matrix" (Tensor): Transformation matrix for the CAV.
                    - "pred_box" (Tensor): Predicted bounding boxes from the CAV.
                    - "pred_score" (Tensor): Prediction scores for the bounding boxes.
                    - "confidence_map" (Tensor): Confidence map for the predicted results.
                - "fusion_box" (Tensor): The fused bounding box.
                - "sence_id" (int): The sence ID.
                - "fusion_future" (Tensor): The future bounding box.
                - "anchor_box" (Tensor): The anchor box.
            detector (Detector): The detector object used for prediction.
        Returns:
            list: A list of filtered unmatched bounding boxes that do not match the
                  fused bounding box and have confidence scores above the threshold.
        """
        result = {"remove": [], "spoof": [], 'remove_score': [], 'spoof_score': []}
        sence_id = case["sence_id"]
        scenario_index = case["scenario_index"]
        fusion_feature = case["fusion_feature"]
        fusion_output = case["fusion_output"]
        fusion_boxes = fusion_output["pred_box_tensor"]
        # fusion_boxes_center = box_utils.corner_to_center_torch(fusion_boxes, order=detector.post_processor.params['order'])
        fusion_score = fusion_output["pred_score"]

        data_dict = self.time_series_dataset.__getitem__(sence_id - (scenario_index+1)*self.condition_frame)
        batch_data = self.time_series_dataset.collate_fn([data_dict])
        batch_data = train_utils.to_device(batch_data, self.device)

        anchor_box = case['fusion_anchor_box']
        batch_data["anchor_box"] = anchor_box
        

        with torch.no_grad():
            start_time = time.time()
            output_dict = self.worldmodel.forward(batch_data)
            end_time = time.time()
            result['world_model_time'] = end_time - start_time
            pre_feature = output_dict["pred_features"]
            start_time = time.time()
            prob = F.sigmoid(output_dict["psm"].permute(0, 2, 3, 1)).reshape(-1)
            proposals = VoxelPostprocessor.delta_to_boxes3d(
                output_dict["rm"], anchor_box
            )[0]
            proposals = box_utils.boxes_to_corners_3d(proposals, order=detector.post_processor.params['order'])
            pre_boxes, pre_score = detector.post_processor.post_process_single_car(batch_data, output_dict)
            if len(pre_boxes) == 0:
                print("error:",sence_id)
                return result
            pre_boxes, pre_score = pre_boxes[0], pre_score[0]
            result["pred_output"] = {'pred_box_tensor': pre_boxes, 'pred_score': pre_score}
            unmatched_boxes = self.find_unmatched_boxes(pre_boxes, fusion_boxes, iou_threshold=0.1)
            result["unmatched_boxes"] = unmatched_boxes
            for remove_index in unmatched_boxes["remove"]:
                remove_box = pre_boxes[remove_index]
                remove_score = pre_score[remove_index]
                score = self.consistent_score(remove_box, proposals, prob)
                # if score > self.consistent_score_threshold:
                result["remove"].append(remove_box)
                result["remove_score"].append(score - self.consistent_score_threshold)
            for spoof_index in unmatched_boxes["spoof"]:
                spoof_box = fusion_boxes[spoof_index]
                spoof_score = fusion_score[spoof_index]
                score = self.consistent_score(spoof_box, proposals, prob)
                # if score < self.consistent_score_threshold:
                result["spoof"].append(spoof_box)
                result["spoof_score"].append(self.consistent_score_threshold-score)
            end_time = time.time()
            result['consistent_time'] = end_time - start_time
        return result
    
    def defense(self, case, detector=None):
        spatial_result = {"remove": [], "spoof": [], 'remove_score': [], 'spoof_score': []}
        temporal_result = {"remove": [], "spoof": [], 'remove_score': [], 'spoof_score': []}
        if self.params["if_spatial"]:
            spatial_result = self.spatial_consistent_check(case, detector=detector)
        if self.params["if_temporal"]:
            temporal_result = self.temporal_consistent_check(case, detector=detector)

        remove_list = spatial_result["remove"] + temporal_result["remove"]
        remove_score = spatial_result["remove_score"] + temporal_result["remove_score"]
        spoof_list = spatial_result["spoof"] + temporal_result["spoof"]
        spoof_score = spatial_result["spoof_score"] + temporal_result["spoof_score"]
        result = { "remove_list": remove_list, "remove_score": remove_score, "spoof_list": spoof_list, "spoof_score": spoof_score, "world_model_time": temporal_result.get("world_model_time", 0), "consistent_time": temporal_result.get("consistent_time", 0)}
        return result
    
    def defense_evaluation_processor(self, result, case, defense_results):
        """
        """
        fusion_boxes = case["fusion_box"]
        gt_bboxes = case["gt_bboxes"]
        cav_position = case["car_position"]
        attack_bbox = case["attack_bbox_corners"]
        remove_list = result["remove_list"]
        remove_score = result["remove_score"]
        spoof_list = result["spoof_list"]
        spoof_score = result["spoof_score"]
        # Convert cav_position (dict) to an array of positions (assumed to be [x,y,z])
        cav_positions = np.array(list(cav_position.values()))  # shape: (num_cav, 3)
        
        is_success = False
        
        if gt_bboxes is None and fusion_boxes is None:
            return defense_results
        elif gt_bboxes is None:
            spoof_label = np.array([True] * fusion_boxes.shape[0])
            remove_label = np.zeros(0)
            remove_error = np.zeros(0)
            remove_mask = np.array([]).astype(bool)
            spoof_error = np.full(fusion_boxes.shape[0], -10)
            
        elif fusion_boxes is None:
            remove_label = np.array([True] * gt_bboxes.shape[0])
            remove_error = np.zeros(gt_bboxes.shape[0])
            spoof_label = np.zeros(0)
            spoof_error = np.zeros(0)
            spoof_mask = np.array([]).astype(bool)
            
        else:
            iou = compute_overlaps(gt_bboxes, fusion_boxes)
            spoof_label = np.max(iou, axis=0) <= self.iou_thres
            remove_label = np.max(iou, axis=1) <= self.iou_thres
            spoof_error = np.full(fusion_boxes.shape[0], -10) 
            remove_error = np.zeros(gt_bboxes.shape[0])

        if fusion_boxes is not None:   
            # Get the x and y coordinates of fusion_boxes (assumed fusion_boxes is a numpy array)
            fusion_centers = fusion_boxes.mean(dim=1)[:, :2].cpu().numpy()  # Compute each bbox's (x, y) center and convert to a numpy array
            # For each fusion box, set spoof_mask True if its distance to every cav_position is greater than 1
            spoof_mask = np.array([
                np.all(np.linalg.norm(cav_positions[:, :2] - fc, axis=1) > 1)
                for fc in fusion_centers
            ])
            
        if gt_bboxes is not None:
            # Get the x and y coordinates of fusion_boxes (assumed fusion_boxes is a numpy array)
            gt_centers = gt_bboxes.mean(dim=1)[:, :2].cpu().numpy()  # Compute the center (x, y) from 8 vertices per box
            # For each fusion box, set spoof_mask True if its distance to every cav_position is greater than 1
            remove_mask = np.array([
                np.all(np.linalg.norm(cav_positions[:, :2] - fc, axis=1) > 1)
                for fc in gt_centers
            ])

        for s_box, s_score in zip(spoof_list, spoof_score):
            iou_val = compute_iou(s_box, fusion_boxes)
            if iou_val.max() > 0.3:
                idx = np.argmax(iou_val)
                if s_score > spoof_error[idx]:
                    spoof_error[idx] = s_score

        for r_box, r_score in zip(remove_list, remove_score):
            iou_val = compute_iou(r_box, gt_bboxes)
            if iou_val.max() > self.iou_thres:
                idx = np.argmax(iou_val)
                if r_score > remove_error[idx]:
                    remove_error[idx] = r_score
                    
        remove_iou = compute_iou(attack_bbox, remove_list)
        spoof_iou = compute_iou(attack_bbox, spoof_list)
        if remove_iou.max() > self.iou_thres or spoof_iou.max() > 0.3:
            is_success = True
        if self.params["if_mask"]:
            defense_results["spoof_error"].append(spoof_error[spoof_mask])
            defense_results["spoof_label"].append(spoof_label[spoof_mask])
            defense_results["remove_error"].append(remove_error[remove_mask])
            defense_results["remove_label"].append(remove_label[remove_mask])
        else:
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

        return best_TPR, best_FPR, roc_auc, best_thres , tpr_data, fpr_data

        
            


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_file = os.path.join(
        uap_path, "configs/uap.yaml"
    )
    config = yaml_utils.load_yaml(config_file, None)
    defender = spatial_temporal_defender(config=config, device=device)
    sence_id = 1717
    
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
    # single_cav_anchor_box = torch.from_numpy(
    #     single_cav_detector.post_processor.generate_anchor_box()
    # ).to(device)

    cav_content = v2v_batch_data["ego"]
 
    # cav_content["lidar_pose"] = np.array(cav_content["lidar_pose"])

    obj_index_dict = get_random_index(cav_content, v2v_dataset, 1)
    obj_id = list(obj_index_dict.keys())[0]
    feature_index, obj_bbox = obj_index_dict[obj_id]
    
    data_dict = fusion_detector.feature_encoder(cav_content)
    data_dict["anchor_box"] = fusion_anchor_box
    
    attack_mode = config['eval_params']['attack_mode']
    attack_dict = {"attack_tagget": attack_mode}
    attack_dict["obj_idx"] = feature_index
    attack_dict["object_bbox"] = obj_bbox
    # print(attack_dict)
    output_dict = uap.attack(data_dict, fusion_detector, attack_dict=attack_dict, apply_attack=True)
    obj_bbox = box_utils.boxes_to_corners_3d(obj_bbox.unsqueeze(0), order=fusion_detector.post_processor.params['order'])[0]
    # data_dict["anchor_box"] = single_cav_anchor_box
    single_cav_output = uap.attack_single_car(data_dict, single_cav_detector, attack_dict=attack_dict, apply_attack=False)
    cav_position = cav_content['cav_pose']
    iou_attack = compute_iou(obj_bbox, output_dict["pred_box_tensor"])
    if len(iou_attack)>0 and iou_attack.max() > 0.1:
        print("attack failed")
    
    sence_case = {
        'sence_id': sence_id,
        'scenario_index': scenario_id,
        'fusion_feature': output_dict['fused_feature'],
        'fusion_anchor_box': fusion_anchor_box,
        'fusion_output': output_dict,
        "gt_bboxes": gt_box_tensor,
        "attack_bbox_corners": obj_bbox,
        "fusion_box": output_dict['pred_box_tensor'],
        'single_cav_anchor_box': fusion_anchor_box,
        'single_cav_output': single_cav_output,
        'car_position': cav_position
        }
    defense_results = {
            "spoof_error": [],
            "spoof_label": [],
            # "spoof_location": [],
            "remove_error": [],
            "remove_label": [],
            # "remove_location": [],
            "success": [],
    }
    result = defender.defense(sence_case, fusion_detector)
    defense_results = defender.defense_evaluation_processor(result, sence_case, defense_results)
    for key, data in defense_results.items():
        defense_results[key] = np.concatenate(data).reshape(-1)
    if attack_mode == "remove":
        best_TPR, best_FPR, roc_auc, best_thres = defender.compute_roc(defense_results["remove_error"], defense_results["remove_label"])
    elif attack_mode == "spoof":
        best_TPR, best_FPR, roc_auc, best_thres = defender.compute_roc(defense_results["spoof_error"], defense_results["spoof_label"])
    print("AUC: ", roc_auc)
    print("TPR: ", best_TPR)
    print("FPR: ", best_FPR)
    print("Threshold: ", best_thres)
    print("Success: ", np.mean(defense_results["success"]))

        
