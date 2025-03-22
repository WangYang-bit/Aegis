import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from opencood.utils import box_utils

from mvp.data.util import bbox_sensor_to_map, pcd_sensor_to_map
from mvp.visualize.general import get_xylims


def draw_attack(data_dict, attack_bboxes, normal_bboxes=None, show=False, save=None):
    fig, axes = plt.subplots(1, 2, figsize=(40, 20))

    ax_normal = axes[0]
    ax_attack = axes[1]
    # draw point clouds
    # pointcloud_all = pcd_sensor_to_map(case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar"], case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar_pose"])[:,:3]
    pointcloud_all = pcd_sensor_to_map(
        data_dict["origin_lidar"], data_dict["lidar_pose"]
    )[:, :3]
    xlim, ylim = get_xylims(pointcloud_all)
    ax_normal.set_xlim(xlim)
    ax_normal.set_ylim(ylim)
    ax_attack.set_xlim(xlim)
    ax_attack.set_ylim(ylim)
    # ax.set_aspect('equal', adjustable='box')
    ax_normal.scatter(pointcloud_all[:, 0], pointcloud_all[:, 1], s=0.01, c="black")
    ax_attack.scatter(pointcloud_all[:, 0], pointcloud_all[:, 1], s=0.01, c="black")

    # draw gt/result bboxes
    attack_total_bboxes = []
    normal_total_bboxes = []
    attack_target_bboxes = []
    if "gt_bboxes" in data_dict:
        gt_bboxes = data_dict["gt_bboxes"].cpu().numpy()
        gt_bboxes = box_utils.corner_to_center(gt_bboxes, order="hwl")
        attack_total_bboxes.append(
            (
                bbox_sensor_to_map(gt_bboxes, data_dict["lidar_pose"]),
                data_dict["object_ids"],
                "g",
            )
        )
        normal_total_bboxes.append(
            (
                bbox_sensor_to_map(gt_bboxes, data_dict["lidar_pose"]),
                data_dict["object_ids"],
                "g",
            )
        )
    if "attack_bbox" in data_dict:
        attack_bbox = data_dict["attack_bbox"].cpu().numpy()
        attack_target_bboxes.append(
            (
                np.array([bbox_sensor_to_map(attack_bbox, data_dict["lidar_pose"])]),
                data_dict["object_ids"],
                "b",
            )
        )
        
    if attack_bboxes is not None:
        attack_bboxes = attack_bboxes.cpu().numpy()
        attack_bboxes = box_utils.corner_to_center(attack_bboxes, order="hwl")
        attack_total_bboxes.append(
            (bbox_sensor_to_map(attack_bboxes, data_dict["lidar_pose"]), None, "r")
        )
    if normal_bboxes is not None:
        normal_bboxes = normal_bboxes.cpu().numpy()
        normal_bboxes = box_utils.corner_to_center(normal_bboxes, order="hwl")
        normal_total_bboxes.append(
            (bbox_sensor_to_map(normal_bboxes, data_dict["lidar_pose"]), None, "r")
        )

    draw_bbox_2d(ax_attack, attack_total_bboxes)
    draw_bbox_2d(ax_normal, normal_total_bboxes)
    draw_bbox_2d(ax_attack, attack_target_bboxes)
    draw_bbox_2d(ax_normal, attack_target_bboxes)

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
        print("save to ", save)
    plt.close()


def draw_point_cloud(data_dict, show=False, save=None):
    fig, ax = plt.subplots(1, 1, figsize=(40, 20))

    # draw point clouds
    # pointcloud_all = pcd_sensor_to_map(case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar"], case[frame_id][attack["attack_opts"]["attacker_vehicle_id"]]["lidar_pose"])[:,:3]
    pointcloud_all = pcd_sensor_to_map(
        data_dict["origin_lidar"], data_dict["lidar_pose"]
    )[:, :3]
    xlim, ylim = get_xylims(pointcloud_all)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # ax.set_aspect('equal', adjustable='box')
    ax.scatter(pointcloud_all[:, 0], pointcloud_all[:, 1], s=0.01, c="black")

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()


def draw_multi_pointclouds(pointclouds, show=True, save=None):
    fig, ax = plt.subplots(1, 1, figsize=(40, 20))
    colormap = plt.get_cmap("tab10")
    color_cache = {}

    for pc in pointclouds:
        x, y, z, label = pc[:, 0], pc[:, 1], pc[:, 2], pc[:, 3]
        unique_labels = np.unique(label)

        for lbl in unique_labels:
            if lbl not in color_cache:
                color_cache[lbl] = colormap(len(color_cache) % 10)
            indices = label == lbl
            ax.scatter(
                x[indices],
                y[indices],
                s=0.01,
                color=color_cache[lbl],
                label=f"Class {int(lbl)}",
            )

    # 如果想避免重复图例，可加这一行:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    if show:
        plt.show()
    if save is not None:
        plt.savefig(save)
    plt.close()


def draw_bbox_2d(ax, bboxes_id_color):
    for bboxes, bboxes_ids, color in bboxes_id_color:
        for i in range(bboxes.shape[0]):
            boxp = cv2.boxPoints(
                (
                    (bboxes[i][0], bboxes[i][1]),
                    (bboxes[i][5], bboxes[i][4]),
                    bboxes[i][6] / np.pi * 180,
                )
            )
            boxp = np.insert(boxp, boxp.shape[0], boxp[0, :], 0)
            xs, ys = zip(*boxp)
            ax.plot(xs, ys, linewidth=1, color=color)
            if bboxes_ids is not None:
                ax.text(xs[0], ys[0], str(bboxes_ids[i]), fontsize="xx-small")


def draw_bbox(pre_bboxes, gt_bboxes, save=None):
    # Project each corner onto x-y plane and plot with different colors
    ax = plt.gca()

    for box in pre_bboxes:
        pts = box[:, :2]
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], color="red")

    for box in gt_bboxes:
        pts = box[:, :2]
        pts = np.vstack([pts, pts[0]])
        ax.plot(pts[:, 0], pts[:, 1], color="green")
    if save is not None:
        plt.savefig(save)


def get_xylims(points):
    xlim, ylim = (
        [points[:, 0].min(), points[:, 0].max()],
        [points[:, 1].min(), points[:, 1].max()],
    )
    lim = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    xlim = [sum(xlim) / 2 - lim / 2, sum(xlim) / 2 + lim / 2]
    ylim = [sum(ylim) / 2 - lim / 2, sum(ylim) / 2 + lim / 2]
    return xlim, ylim


def draw_cls_score(output_dict, save=None):
    """
    Visualize the classification scores as a heatmap.

    Args:
        classification_scores (torch.Tensor): The classification scores tensor of shape (H, W, 2).

    Returns:
        None
    """
    file_name = "cls_score.png"
    save = os.path.join(save, file_name)
    prob = output_dict["psm"]
    prob = F.sigmoid(prob.permute(0, 2, 3, 1))
    classification_scores = prob[0].cpu()
    # Convert tensor to numpy array
    scores_np = classification_scores.numpy()

    # Compute the mean along the last dimension
    mean_scores = scores_np.mean(axis=-1)
    # mean_scores = mean_scores.T

    # Create the heatmap
    plt.figure(figsize=(64, 32))
    plt.imshow(mean_scores, cmap="coolwarm", interpolation="nearest")

    # Add text annotations
    for i in range(mean_scores.shape[0]):
        for j in range(mean_scores.shape[1]):
            plt.text(
                j,
                i,
                f"{mean_scores[i, j]:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=6,
            )

    # Add color bar
    plt.colorbar()

    if save is not None:
        plt.savefig(save)
        print("save to ", save)
    plt.close()


# Function to project 3D points to a plane
def do_range_projection(points, proj_fov_up, proj_fov_down, proj_W, proj_H):
    fov_up = proj_fov_up / 180.0 * np.pi  # 转换为弧度
    fov_down = proj_fov_down / 180.0 * np.pi  # 转换为弧度
    fov = abs(fov_down) + abs(fov_up)  # 总视场角

    # 计算所有点的深度
    depth = np.linalg.norm(points, 2, axis=1)

    # 获取扫描组件
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]

    # 计算所有点的角度
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # 投影到图像坐标
    proj_x = 0.5 * (yaw / np.pi + 1.0)  # [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # [0.0, 1.0]

    # 缩放到图像大小
    proj_x_img = proj_x * proj_W  # [0.0, W]
    proj_y_img = proj_y * proj_H  # [0.0, H]

    return proj_x, proj_y, proj_x_img, proj_y_img


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate a sample 3D point cloud (e.g., a sphere)
    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 60)
    phi, theta = np.meshgrid(phi, theta)
    r = 1.0
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T

    # Laser scan parameters
    proj_fov_up = 2.0  # degrees
    proj_fov_down = -30.0  # degrees
    proj_W = 512
    proj_H = 64
    # 执行投影
    proj_x, proj_y, proj_x_img, proj_y_img = do_range_projection(
        points, proj_fov_up, proj_fov_down, proj_W, proj_H
    )

    # 根据 proj_x 值为点赋予颜色
    colors = proj_x
    colors = (colors - colors.min()) / (colors.max() - colors.min())

    # 可视化原始3D点云，颜色根据 proj_x 值
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    sc = ax1.scatter(
        points[:, 0], points[:, 1], points[:, 2], c=colors, cmap="hsv", s=1
    )
    ax1.set_title("带有 proj_x 颜色的原始3D点云")
    fig.colorbar(sc, ax=ax1, label="proj_x")

    # 可视化投影后的2D图像，颜色根据 proj_x 值
    ax2 = fig.add_subplot(122)
    sc2 = ax2.scatter(proj_x_img, proj_y_img, c=colors, cmap="hsv", s=1)
    ax2.set_title("带有 proj_x 颜色的投影2D图像")
    ax2.set_xlim(0, proj_W)
    ax2.set_ylim(0, proj_H)
    ax2.invert_yaxis()  # 反转y轴以匹配图像坐标
    fig.colorbar(sc2, ax=ax2, label="proj_x")

    # plt.show()
    plt.savefig("range_projection.png")
