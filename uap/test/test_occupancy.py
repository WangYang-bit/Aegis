import os
import sys

root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
sys.path.append(root)
import pickle

import matplotlib.pyplot as plt
import numpy as np
from config import (
    class_id_inv_map,
    data_root,
    model_root,
    scenario_maps,
    uap_path,
    uap_root,
)
from torch.utils.data import DataLoader

from mvp.data.util import pcd_sensor_to_map
from mvp.defense.detection_util import filter_segmentation
from mvp.tools.ground_detection import get_ground_plane
from mvp.tools.polygon_space import get_free_space, get_occupied_space
from mvp.tools.squeezeseg.interface import SqueezeSegInterface

sys.path.append(uap_root)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset

from uap.tools.feature_analys import *
from uap.utils import train_utils

# from uap.tools.squeezeseg.interface import SqueezeSegInterface
# from uap.data.utils import pcd_sensor_to_map
# from uap.tools.polygon_space import get_occupied_space, get_free_space, points_to_polygon
# from uap.tools.ground_detection import get_ground_plane

result_path = os.path.join(uap_path, "test/result")
os.makedirs(result_path, exist_ok=True)
attack_config_file = os.path.join(uap_path, "configs/uap.yaml")

train_path = os.path.join(data_root, "train")
test_path = os.path.join(data_root, "test")
squeeze_data_root = os.path.join(uap_root, "data")

# def filter_segmentation(pcd, lidar_seg, lidar_pose, in_lane_mask=None, point_height=None, max_range=50):
#     object_segments = []
#     for info in lidar_seg["info"]:
#         object_points = pcd[info["indices"]]
#         if np.min(np.sum(object_points[:,:2] ** 2, axis=1)) > max_range ** 2:
#             continue
#         object_points = pcd_sensor_to_map(object_points, lidar_pose)
#         if point_height is not None:
#             if point_height[info["indices"]].min() < -0.5 or point_height[info["indices"]].max() > 3 or point_height[info["indices"]].max() < 0.6:
#                 continue
#         if scipy.spatial.distance.cdist(object_points[:, :2], object_points[:, :2]).max() > 8:
#             continue
#         occupied_area = points_to_polygon(object_points[:,:2])
#         if occupied_area.area > 20:
#             continue
#         if occupied_area.area < 0.5:
#             continue
#         if in_lane_mask is not None:
#             if in_lane_mask[info["indices"]].sum() <= 0.2 * len(info["indices"]):
#                 continue
#         object_segments.append(info["indices"])
#     return object_segments


def project_to_grid(pcd, point_class, point_score, grid_size=0.1):
    x, y = pcd[:, 0], pcd[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_bins = int((x_max - x_min) / grid_size) + 1
    y_bins = int((y_max - y_min) / grid_size) + 1

    # Project each point onto the grid
    ix = ((x - x_min) / grid_size).astype(int)
    iy = ((y - y_min) / grid_size).astype(int)

    # Collect the class and score for each grid
    grid_data = {}
    for i in range(len(pcd)):
        key = (ix[i], iy[i])
        if key not in grid_data:
            grid_data[key] = []

        grid_data[key].append((point_class[i], point_score[i]))

    # Compute the most frequent class and average score for each grid
    type_grid = np.zeros((y_bins, x_bins), dtype=int)
    score_grid = np.zeros((y_bins, x_bins), dtype=float)

    for (gx, gy), vals in grid_data.items():
        classes = [v[0] for v in vals]
        scores = [v[1] for v in vals]
        # Most frequent class
        unique, counts = np.unique(classes, return_counts=True)
        major_class = unique[np.argmax(counts)]
        # Average score
        # mean_score = np.mean([s for (c, s) in vals if c == major_class])
        mean_score = np.mean(scores)
        if mean_score > 0.5 and major_class == 0:
            type_grid[gy, gx] = 2
            score_grid[gy, gx] = mean_score
            continue
        type_grid[gy, gx] = major_class
        score_grid[gy, gx] = mean_score

    # Visualization
    fig, axs = plt.subplots(1, 2, figsize=(40, 16))
    im1 = axs[0].imshow(type_grid, origin="lower", cmap="jet")
    axs[0].set_title("Type Grid")
    plt.colorbar(im1, ax=axs[0])

    im2 = axs[1].imshow(score_grid, origin="lower", cmap="viridis")
    axs[1].set_title("Score Grid")
    plt.colorbar(im2, ax=axs[1])

    plt.tight_layout()
    plt.savefig(os.path.join(result_path, "grid.png"))


def visualize_polygons(occupied_areas, free_areas):
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
    plt.savefig(os.path.join(result_path, "polygons.png"))


def test_occupancy_map(case, lidar_seg_api):
    lidar, lidar_pose = case["lidar"], case["lidar_pose"]
    pcd = pcd_sensor_to_map(lidar, lidar_pose)

    lane_info = pickle.load(
        open(
            os.path.join(
                squeeze_data_root, "carla/{}_lane_info.pkl".format(case["map"])
            ),
            "rb",
        )
    )
    lane_areas = pickle.load(
        open(
            os.path.join(
                squeeze_data_root, "carla/{}_lane_areas.pkl".format(case["map"])
            ),
            "rb",
        )
    )
    lane_planes = pickle.load(
        open(
            os.path.join(
                squeeze_data_root, "carla/{}_ground_planes.pkl".format(case["map"])
            ),
            "rb",
        )
    )

    ground_indices, in_lane_mask, point_height = get_ground_plane(
        pcd,
        lane_info=lane_info,
        lane_areas=lane_areas,
        lane_planes=lane_planes,
        method="map",
    )
    lidar_seg = lidar_seg_api.run(lidar)

    point_class = lidar_seg["class"]
    point_score = lidar_seg["score"]

    # # project_to_grid(pcd, point_class, point_score, 1)

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
    project_to_grid(lidar, new_point_class, point_score, 0.4)
    # ego_bbox = case["ego_bbox"]
    # ego_area = bbox_to_polygon(ego_bbox)
    # ego_area_height = ego_bbox[5]

    ret = {
        # "ego_area": ego_area,
        # "ego_area_height": ego_area_height,
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
    ret["occupied_areas"] = occupied_areas
    ret["occupied_areas_height"] = occupied_areas_height
    ret["free_areas"] = free_areas

    return ret


if __name__ == "__main__":
    model_dir = os.path.join(model_root, "pointpillar_single_car/config.yaml")

    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = train_path
    hypes["validate_dir"] = test_path
    dataset = build_dataset(hypes, visualize=True, train=False)
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
    sence_id = 741
    # for i, batch_data in tqdm(enumerate(data_loader)):
    #     if i == sence_id :
    #         break
    data_dict = dataset.__getitem__(sence_id)
    batch_data = dataset.collate_batch_test([data_dict])
    batch_data = train_utils.to_device(batch_data, device)

    # gt_box_tensor, object_ids = dataset.post_processor.generate_gt_bbx(batch_data)
    cav_content = batch_data["ego"]

    cav_content["lidar"] = cav_content["lidar"].cpu().numpy()
    cav_content["lidar_pose"] = np.array(cav_content["lidar_pose"])

    lidar_seg_api = SqueezeSegInterface()
    sence_map = scenario_maps[cav_content["scenario_id"]]
    case = {
        "lidar": cav_content["lidar"],
        "lidar_pose": cav_content["lidar_pose"],
        "map": sence_map,
    }
    omap = test_occupancy_map(case, lidar_seg_api)
    occupied_areas = omap["occupied_areas"]
    free_areas = omap["free_areas"]
    visualize_polygons(occupied_areas, free_areas)
