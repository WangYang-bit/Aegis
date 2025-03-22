import os
import sys
import time

from tqdm import tqdm

root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
sys.path.append(root)
import pickle

import matplotlib.pyplot as plt
import numpy as np
from uap.config import (
    class_id_inv_map,
    data_root,
    model_root,
    scenario_maps,
    uap_path,
    uap_root,
)
from torch.utils.data import DataLoader

from mvp.data.util import pcd_sensor_to_map, get_distance, bbox_sensor_to_map, bbox_map_to_sensor
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.polygon_space import get_free_space, get_occupied_space, bbox_to_polygon
from mvp.tools.squeezeseg.interface import SqueezeSegInterface
from mvp.tools.iou import iou3d
sys.path.append(uap_root)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import box_utils

from uap.tools.feature_analys import *
from uap.utils import train_utils

from uap.defense.base_defender import BaseDefender
from uap.attaker import UniversalAttacker
from mvp.defense.perception_defender import PerceptionDefender

class mvp_defender(BaseDefender):
    def __init__(self, device = "cuda"):
        self.name = "mvp_defender"
        self.device = device
        self.defender = PerceptionDefender()
        self.lidar_seg_api = SqueezeSegInterface()
        train_path = os.path.join(data_root, "train")
        test_path = os.path.join(data_root, "test")
        self.squeeze_data_root = os.path.join(uap_root, "data")

        model_dir = os.path.join(model_root, "pointpillar_single_car/config.yaml")

        hypes = yaml_utils.load_yaml(model_dir, None)
        hypes["root_dir"] = train_path
        hypes["validate_dir"] = test_path
        self.single_car_dataset = build_dataset(hypes, visualize=False, train=False)
        self.occupancy_map_root = "/root/autodl-tmp/OPV2V/occupy_map"

    def defense(self, case):
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
        fusion_box = case["fusion_box"]
        if fusion_box is None:
            return None
        else:
            fusion_box = fusion_box.cpu().numpy()
            fusion_box = box_utils.corner_to_center(fusion_box, order="hwl")
        sence_id = case["sence_id"]
        victim_id = 1
        victim_cav_id = None
        # start_time = time.time()
        data_dict = self.single_car_dataset.__getitem__(sence_id)
        batch_data = self.single_car_dataset.collate_batch_test([data_dict])
        batch_data = train_utils.to_device(batch_data, self.device)
        end_time = time.time()
        # print("Data loading time: ", end_time - start_time)
        occupancy_feature = {}
        for cav_id, cav_content in batch_data.items():
            cav_content["lidar"] = cav_content["lidar"].cpu().numpy()
            cav_content["lidar_pose"] = np.array(cav_content["lidar_pose"])
            if list(batch_data.keys()).index(cav_id) == victim_id:
                victim_cav_id = cav_id
                cav_content["pred_bboxes"] = bbox_map_to_sensor(bbox_sensor_to_map(fusion_box, case["ego_lidar_pose"]), cav_content["lidar_pose"]) 
                cav_content["pred_bboxes_map"] = bbox_sensor_to_map(fusion_box, case["ego_lidar_pose"])
            
        #     sence_map = scenario_maps[cav_content["scenario_id"]]
        #     occupancy_case = {
        #         "ego_bbox": cav_content["ego_bbox"],
        #         "lidar": cav_content["lidar"],
        #         "lidar_pose": cav_content["lidar_pose"],
        #         "map": sence_map,
        #     }
        #     start_time = time.time()
        #     occupancy_feature[cav_id] = self.test_occupancy_map(occupancy_case, self.lidar_seg_api)
        #     end_time = time.time()
        #     # print("Occupancy map time: ", end_time - start_time)
        # return occupancy_feature
        if victim_cav_id is None:
            return None        

        file_path = os.path.join(f"/root/autodl-tmp/OPV2V/occupy_map/{sence_id}", "occupy_feature.pkl")
        with open(file_path, "rb") as f:
            occupancy_feature = pickle.load(f)
        defense_case = {}
        defense_case[0] = {}
        for cav_id, cav_content in batch_data.items():
            defense_case[0][cav_id] = {}
            defense_case[0][cav_id].update(cav_content)

        for cav_id, cav_content in occupancy_feature.items():
            defense_case[0][cav_id].update(cav_content)
        defense_case, score, metrics = self.defender.run(defense_case, {"frame_ids": [0]})
        vehicle_metrics = metrics[0][victim_cav_id]
        return vehicle_metrics

    def defense_evaluation_processor(self, vehicle_metrics , case, defense_results, iou_thres=0.7, dist_thres=40):
        attack_mode = case["attack_mode"]
        attack_bbox = case["attack_bbox"].cpu().numpy()
        attack_bbox = bbox_sensor_to_map(attack_bbox, case["ego_lidar_pose"])
        gt_bboxes = vehicle_metrics["gt_bboxes"]
        pred_bboxes = vehicle_metrics["pred_bboxes"]
        lidar_pose = vehicle_metrics["lidar_pose"]

        # iou 2d
        gt_bboxes[:, 2] = 0
        gt_bboxes[:, 5] = 1
        pred_bboxes[:, 2] = 0
        pred_bboxes[:, 5] = 1

        iou = np.zeros((gt_bboxes.shape[0], pred_bboxes.shape[0]))
        for i, gt_bbox in enumerate(gt_bboxes):
            for j, pred_bbox in enumerate(pred_bboxes):
                iou[i, j] = iou3d(gt_bbox, pred_bbox)

        spoof_label = np.max(iou, axis=0) <= iou_thres
        spoof_mask = np.logical_and(get_distance(pred_bboxes[:, :2], lidar_pose[:2]) > 1, get_distance(pred_bboxes[:, :2], lidar_pose[:2]) <= dist_thres)
        remove_label = np.max(iou, axis=1) <= iou_thres
        remove_mask = get_distance(gt_bboxes[:, :2], lidar_pose[:2]) <= dist_thres

        spoof_error = np.zeros(pred_bboxes.shape[0])
        spoof_location = np.zeros((pred_bboxes.shape[0], 2))
        for error_area, error, gt_error, bbox_index in vehicle_metrics["spoof"]:
            if error > spoof_error[bbox_index]:
                spoof_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                spoof_error[bbox_index] = error

        remove_error = np.zeros(gt_bboxes.shape[0])
        remove_location = np.zeros((gt_bboxes.shape[0], 2))
        for error_area, error, gt_error, bbox_index in vehicle_metrics["remove"]:
            if bbox_index < 0:
                continue
            if error > remove_error[bbox_index]:
                remove_location[bbox_index] = np.array(list(list(error_area.centroid.coords)[0]))
                remove_error[bbox_index] = error

        detected_location = spoof_location if attack_mode == "spoof" else remove_location
        is_success = np.min(get_distance(detected_location, attack_bbox[:2])) < 2

        defense_results["spoof_error"].append(spoof_error[spoof_mask])
        defense_results["spoof_label"].append(spoof_label[spoof_mask])
        defense_results["spoof_location"].append(spoof_location[spoof_mask])
        defense_results["remove_error"].append(remove_error[remove_mask])
        defense_results["remove_label"].append(remove_label[remove_mask])
        defense_results["remove_location"].append(remove_location[remove_mask])
        defense_results["success"].append(np.array([is_success]).astype(np.int8))
        return defense_results

    def test_occupancy_map(self, case, lidar_seg_api):
        lidar, lidar_pose = case["lidar"], case["lidar_pose"]
        pcd = pcd_sensor_to_map(lidar, lidar_pose)
        start_time = time.time()
        lane_info = pickle.load(
            open(
                os.path.join(
                    self.squeeze_data_root, "carla/{}_lane_info.pkl".format(case["map"])
                ),
                "rb",
            )
        )
        lane_areas = pickle.load(
            open(
                os.path.join(
                    self.squeeze_data_root, "carla/{}_lane_areas.pkl".format(case["map"])
                ),
                "rb",
            )
        )
        lane_planes = pickle.load(
            open(
                os.path.join(
                    self.squeeze_data_root, "carla/{}_ground_planes.pkl".format(case["map"])
                ),
                "rb",
            )
        )
        end_time = time.time()
        # print("Map loading time: ", end_time - start_time)
        ground_indices, in_lane_mask, point_height = get_ground_plane(
            pcd,
            lane_info=lane_info,
            lane_areas=lane_areas,
            lane_planes=lane_planes,
            method="map",
        )
        start_time = time.time()
        lidar_seg = lidar_seg_api.run(lidar)
        end_time = time.time()
        # print("Lidar seg time: ", end_time - start_time)

        point_class = lidar_seg["class"]
        point_score = lidar_seg["score"]

        # # project_to_grid(pcd, point_class, point_score, 1)
        start_time = time.time()
        new_point_class = np.zeros(pcd.shape[0])
        object_segments = filter_segmentation(
            lidar,
            lidar_seg,
            lidar_pose,
            in_lane_mask=in_lane_mask,
            point_height=point_height,
            max_range=50,
        )
        object_mask = np.zeros(pcd.shape[0]).astype(bool)
        if len(object_segments) > 0:
            object_indices = np.hstack(object_segments)
            object_mask[object_indices] = True
            new_point_class[object_mask == True] = class_id_inv_map["car"]
        ego_bbox = case["ego_bbox"]
        ego_area = bbox_to_polygon(ego_bbox)
        ego_area_height = ego_bbox[5]

        ret = {
            "ego_area": ego_area,
            "ego_area_height": ego_area_height,
            "plane": None,
            "ground_indices": ground_indices,
            "point_height": point_height,
            "object_segments": object_segments,
        }

        height_thres = 0
        occupied_areas, occupied_areas_height = get_occupied_space(
            pcd, object_segments, point_height=point_height, height_thres=height_thres
        )
        free_areas = get_free_space(
            lidar,
            lidar_pose,
            object_mask,
            in_lane_mask=in_lane_mask,
            point_height=point_height,
            max_range=50,
            height_thres=height_thres,
            height_tolerance=0.2,
        )
        end_time = time.time()
        # print("posprocess time: ", end_time - start_time)
        ret["occupied_areas"] = occupied_areas
        ret["occupied_areas_height"] = occupied_areas_height
        ret["free_areas"] = free_areas

        return ret

    def visualize_polygons(self, occupied_areas, free_areas, save_path=None):
        fig, ax = plt.subplots(figsize=(30, 30))

        # Draw occupied polygons in red
        for poly in occupied_areas:
            x, y = poly.exterior.coords.xy
            ax.fill(x, y, color="red", alpha=0.8, label="Occupied")

        # Draw free polygons in green
        for poly in free_areas:
            x, y = poly.exterior.coords.xy
            ax.fill(x, y, color="green", alpha=0.2, label="Free")

        # Create legend (to avoid duplicate labels, use set())
        handles_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(handles_labels[1], handles_labels[0]))
        ax.legend(by_label.values(), by_label.keys())

        plt.axis("equal")
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.savefig(os.path.join(result_path, "polygons.png"))

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
    result_path = os.path.join(uap_path, "test/result")
    os.makedirs(result_path, exist_ok=True)
    config_file = os.path.join(
        uap_path, "configs/uap.yaml"
    )
    config = yaml_utils.load_yaml(config_file, None)

    train_path = os.path.join(data_root, "train")
    test_path = os.path.join(data_root, "test")

    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")

    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = train_path
    hypes["validate_dir"] = test_path
    dataset = build_dataset(hypes, visualize=False, train=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    uap = UniversalAttacker(config, device)
    fusion_detector = uap.detectors["pointpillar_V2VAM"]
    occupy_feature_dict = {}
    defender = mvp_defender()

    # for sence_id, batch_data in tqdm(enumerate(data_loader)):
    #     if sence_id < 1723:
    #         continue
    #     v2v_batch_data = train_utils.to_device(batch_data, device)
    #     gt_bboxes_tensor, object_ids = dataset.post_processor.generate_gt_bbx(v2v_batch_data)

    #     gt_bboxes = gt_bboxes_tensor.cpu().numpy()
    #     gt_bboxes_center = box_utils.corner_to_center(gt_bboxes, order="hwl")
    #     gt_bboxes_map = bbox_sensor_to_map(gt_bboxes_center, np.array(v2v_batch_data["ego"]["lidar_pose"]))

    #     fusion_anchor_box = torch.from_numpy(
    #         fusion_detector.post_processor.generate_anchor_box()
    #     ).to(device)

    #     cav_content = v2v_batch_data["ego"]
    #     obj_index_dict = get_random_index(cav_content, dataset, 1)
    #     obj_id = list(obj_index_dict.keys())[0]
    #     feature_index, obj_bbox = obj_index_dict[obj_id]
        
    #     data_dict = fusion_detector.feature_encoder(cav_content)
    #     data_dict["anchor_box"] = fusion_anchor_box
    #     attack_mode = config['eval_params']['attack_mode']
    #     attack_dict = {"attack_tagget": attack_mode}
    #     attack_dict["obj_idx"] = feature_index
    #     attack_dict["object_bbox"] = obj_bbox

    #     # print(attack_dict)
    #     output_dict = uap.attack(data_dict, fusion_detector, attack_dict=attack_dict, apply_attack=True)
    #     # gt_box_tensor, object_ids = dataset.post_processor.generate_gt_bbx(batch_data)

    #     scense_case = {
    #         "data_dict": data_dict,
    #         "fusion_box": output_dict['pred_box_tensor'],
    #         "sence_id": sence_id,
    #         "ego_lidar_pose": cav_content["lidar_pose"],
    #         "gt_bboxes": gt_bboxes_map,
    #         "attack_mode" : attack_mode,
    #         "attack_bbox": obj_bbox
    #     }
    #     occupy_feature =  defender.defense(scense_case)
    #     scene_dir = os.path.join("/root/autodl-tmp/OPV2V/occupy_map", str(sence_id))
    #     os.makedirs(scene_dir, exist_ok=True)
    #     save_path = os.path.join(scene_dir, "occupy_feature.pkl")
    #     with open(save_path, "wb") as f:
    #         pickle.dump(occupy_feature, f)
    
    sence_id = 1723
    data_dict = dataset.__getitem__(sence_id)
    v2v_batch_data = dataset.collate_batch_test([data_dict])
    v2v_batch_data = train_utils.to_device(v2v_batch_data, device)
    gt_bboxes_tensor, object_ids = dataset.post_processor.generate_gt_bbx(v2v_batch_data)

    gt_bboxes = gt_bboxes_tensor.cpu().numpy()
    gt_bboxes_center = box_utils.corner_to_center(gt_bboxes, order="hwl")
    gt_bboxes_map = bbox_sensor_to_map(gt_bboxes_center, np.array(v2v_batch_data["ego"]["lidar_pose"]))

    fusion_anchor_box = torch.from_numpy(
        fusion_detector.post_processor.generate_anchor_box()
    ).to(device)

    cav_content = v2v_batch_data["ego"]

    # cav_content["lidar_pose"] = np.array(cav_content["lidar_pose"])

    obj_index_dict = get_random_index(cav_content, dataset, 1)
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
    # gt_box_tensor, object_ids = dataset.post_processor.generate_gt_bbx(batch_data)
    defender = mvp_defender()

    scense_case = {
        "data_dict": data_dict,
        "fusion_box": output_dict['pred_box_tensor'],
        "sence_id": sence_id,
        "ego_lidar_pose": cav_content["lidar_pose"],
        "gt_bboxes": gt_bboxes_map,
        "attack_mode" : attack_mode,
        "attack_bbox": obj_bbox
    }
    defense_results = {
            "spoof_error": [],
            "spoof_label": [],
            "spoof_location": [],
            "remove_error": [],
            "remove_label": [],
            "remove_location": [],
            "success": [],
    }
    start_time = time.time()
    vehicle_metrics =  defender.defense(scense_case)
    end_time = time.time()
    print("Defense time: ", end_time
          - start_time)
    defense_results = defender.defense_evaluation_processor(vehicle_metrics, scense_case, defense_results)
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