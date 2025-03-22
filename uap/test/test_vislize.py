import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

from uap.config import data_root, uap_path, uap_root
from uap.tools.feature_analys import *

root = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../")
sys.path.append(root)


sys.path.append(uap_root)


# from uap.tools.squeezeseg.interface import SqueezeSegInterface

# from uap.tools.polygon_space import get_occupied_space, get_free_space, points_to_polygon
# from uap.tools.ground_detection import get_ground_plane

result_path = os.path.join(uap_path, "test/result")
os.makedirs(result_path, exist_ok=True)
attack_config_file = os.path.join(uap_path, "configs/uap.yaml")

train_path = os.path.join(data_root, "train")
test_path = os.path.join(data_root, "test")
squeeze_data_root = os.path.join(uap_root, "data")


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


if __name__ == "__main__":
    # lidar_file_1 = os.path.join(
    #     SemanticOPV2V_root, "test/2021_08_18_19_48_05/1045/000068_semantic_campos.pcd"
    # )
    # lidar_pose_1 = [
    #     599.7123413085938,
    #     -17.007768630981445,
    #     1.930234432220459,
    #     0.016575083136558533,
    #     -178.533203125,
    #     0.20895595848560333,
    # ]
    # lidar_file_2 = os.path.join(
    #     SemanticOPV2V_root, "test/2021_08_18_19_48_05/1054/000068_semantic_campos.pcd"
    # )
    # lidar_pose_2 = [
    #     550.4852905273438,
    #     -24.436113357543945,
    #     1.9299373626708984,
    #     0.007067767903208733,
    #     179.47259521484375,
    #     0.2185523808002472,
    # ]
    # lidar_file_3 = (
    #     "/share/SemanticOPV2V-OpenMMLab/OPV2V/test/2021_08_18_19_48_05/1045/000068.pcd"
    # )
    # lidar_pose_3 = [
    #     599.7123413085938,
    #     -17.007768630981445,
    #     1.930234432220459,
    #     0.016575083136558533,
    #     -178.533203125,
    #     0.20895595848560333,
    # ]
    # pcd_np_1 = semantic_pcd_to_np(lidar_file_1)
    # pcd_np_1 = pcd_sensor_to_map(pcd_np_1, np.array(lidar_pose_1))
    # pcd_np_2 = semantic_pcd_to_np(lidar_file_2)
    # pcd_np_2 = pcd_sensor_to_map(pcd_np_2, np.array(lidar_pose_2))
    # pcd_np_3 = semantic_pcd_to_np(lidar_file_3)
    # pcd_np_3 = pcd_sensor_to_map(pcd_np_3, np.array(lidar_pose_3))
    # draw_multi_pointclouds(
    #     [pcd_np_1, pcd_np_3],
    #     show=False,
    #     save=os.path.join(result_path, "pcd2.png"),
    # )
    path = "/share/SemanticOPV2V-OpenMMLab/SemanticOPV2V/4LidarCampos/test/2021_08_18_19_48_05/1054/000068_voxels.pth"
    data = torch.load(path)
    print(data)
