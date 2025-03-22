import argparse
import os
from collections import Counter

import numpy as np
import open3d as o3d
import torch
from matplotlib import pyplot as plt
from spconv.pytorch.utils import PointToVoxel

from uap.config import opv2v_label_mapping

class_num = 13
colormap = plt.get_cmap("tab20")
class_color = {i: colormap(i)[:3] for i in range(class_num)}


class Voxelizer:
    def __init__(
        self,
        vsize_xyz,
        coors_range_xyz,
        num_point_features,
        max_num_voxels,
        max_num_points_per_voxel,
        device="cpu",
    ):
        """
        In spconv, PointToVoxel is a class used to convert point clouds into voxel representations.
        The parameters of its constructor are as follows:

        1. vsize_xyz: Defines the size of each voxel. Typically, this is a list or array of length 3,
           representing the size of each voxel along the x, y, and z directions.
           For example, [0.1, 0.1, 0.2] means each voxel has a size of 0.1 along x, 0.1 along y, and 0.2 along z.
           In this case, the voxel sizes along x, y, and z are set to be identical, so only one value is needed,
           such as vsize_xyz = [0.1, 0.1, 0.1].

        2. coors_range_xyz: Specifies the coordinate range used during voxelization.
           This is usually a list or array of length 6, representing the minimum and maximum values for x, y, and z,
           such as [xmin, ymin, zmin, xmax, ymax, zmax]. It defines which points in the point cloud will be converted into voxels.

        3. max_num_points_per_voxel: Indicates the maximum number of points that can be contained in each voxel.
           If the number of points in a voxel exceeds this value, the extra points will be disvehicleded.
           This parameter helps control the voxel size and memory usage.

        4. num_point_features: Specifies the number of features for each point. For example, if each point includes
           x, y, z coordinates and intensity, then the number of features is 4.
           This parameter defines the feature dimension of each point in the point cloud data.

        5. max_num_voxels: Specifies the maximum number of voxels that can be generated during voxelization.
           This is an upper limit to the number of voxels, helping to control computational load.

        6. device: Specifies the device on which the calculation will be performed, usually CPU or GPU.
           This can be a string like "cpu" or "cuda".

        These parameters collectively determine how the point cloud is divided into voxels and represented in a voxel grid.
        """
        self.vsize_xyz = vsize_xyz
        self.voxel_generator = PointToVoxel(
            vsize_xyz=vsize_xyz,
            coors_range_xyz=coors_range_xyz,
            num_point_features=num_point_features,
            max_num_voxels=max_num_voxels,
            max_num_points_per_voxel=max_num_points_per_voxel,
            device=torch.device(device),
        )
        grid_size = np.array(coors_range_xyz[3:]) - np.array(coors_range_xyz[:3])
        self.grid_range = np.round(grid_size / np.array(vsize_xyz)).astype(np.int32)
        self.device = torch.device(device)
        self.voxel_labels = None

    def voxelize(self, point_cloud, label=None):
        """
        1. voxels: This tensor contains the point features within each voxel. The dimensions are typically:
           (max_num_voxels, max_num_points_per_voxel, num_point_features)
           This represents the maximum number of voxels, each containing up to the maximum number of points allowed, with each point having a set number of features.

        2. coordinates: This tensor holds the integer coordinates of each voxel in the grid. The dimensions are:
           (max_num_voxels, 3)
           Each voxel is represented by three coordinates (x, y, z) in the voxel grid.

        3. num_points_per_voxel: This tensor indicates the actual number of points contained in each voxel. The dimensions are:
           (max_num_voxels,)
           This gives a count for each voxel, up to the maximum number of voxels, showing how many points fall within that voxel.

        4. pc_voxel_id: This tensor maps each point in the point cloud to its corresponding voxel ID. The dimensions are:
           (N,)
           Here, N represents the total number of points in the point cloud. This array indicates the voxel ID for each point.
        """
        assert isinstance(point_cloud, np.ndarray), "Point cloud must be a numpy array."
        pc_th = torch.from_numpy(point_cloud).to(self.device).to(torch.float32)
        label_th = torch.from_numpy(label).to(self.device)
        voxels, coordinates, num_points_per_voxel, pc_voxel_id = (
            self.voxel_generator.generate_voxel_with_id(pc_th)
        )
        if label is not None:
            voxel_labels = []
            for voxel_id in range(len(voxels)):
                points_in_voxel = pc_voxel_id == voxel_id
                labels_in_voxel = label_th[points_in_voxel].tolist()

                most_common_label = Counter(labels_in_voxel).most_common(1)[0][0]
                voxel_labels.append(most_common_label)
            voxel_labels = torch.tensor(voxel_labels).cpu()
            self.voxel_labels = voxel_labels

        self.voxels, self.coordinates, self.num_points_per_voxel, self.pc_voxel_id = (
            voxels.cpu(),
            coordinates.cpu(),
            num_points_per_voxel.cpu(),
            pc_voxel_id.cpu(),
        )

        return voxels, coordinates, num_points_per_voxel, pc_voxel_id, self.voxel_labels

    def visualize(self, label=None):
        assert (
            self.coordinates is not None
        ), "No voxel data to visualize. Please run voxelize() first."

        assert self.voxel_labels is not None, "Voxel labels are missing."

        label = self.voxel_labels.numpy() if label is None else label

        voxel_centers = (self.coordinates.numpy() + 0.5) * self.vsize_xyz
        voxel_size = np.array(self.vsize_xyz)
        voxel_meshes = []
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5, origin=[0, 0, 0]
        )

        for idx, center in enumerate(voxel_centers):
            voxel = o3d.geometry.TriangleMesh.create_box(
                width=voxel_size[0], height=voxel_size[1], depth=voxel_size[2]
            )
            voxel.translate(center - voxel_size / 2)

            color = (
                class_color[label[idx]]
                if label is not None and label[idx] in class_color
                else [0.5, 0.5, 0.5]
            )
            voxel.paint_uniform_color(color)
            voxel_meshes.append(voxel)

        o3d.visualization.draw_geometries([mesh_frame, *voxel_meshes])


def find_files(directory, pattern):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(pattern):
                files.append(os.path.join(root, filename))
    return files


def process_pcd_file(
    pcd_file,
    source,
    vsize_xyz,
    coors_range_xyz,
    num_point_features,
    max_num_voxels,
    max_num_points_per_voxel,
    device="cpu",
    cover=True,
):
    voxel_file = pcd_file.replace(source, "voxels.pth")
    if os.path.exists(voxel_file) and not cover:
        return
    voxelizer = Voxelizer(
        vsize_xyz=vsize_xyz,
        coors_range_xyz=coors_range_xyz,
        num_point_features=num_point_features,
        max_num_voxels=max_num_voxels,
        max_num_points_per_voxel=max_num_points_per_voxel,
        device=device,
    )
    semantic_pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(semantic_pcd.points)
    label = (np.asarray(semantic_pcd.colors)[:, 2] * 255).astype(int)
    # Remap the labels using the opv2v_label_mapping
    remapped_label = np.vectorize(opv2v_label_mapping.get)(label)
    voxels, coordinates, num_points_per_voxel, pc_voxel_id, voxel_labels = (
        voxelizer.voxelize(points, label=remapped_label)
    )

    voxel_mat = np.full(voxelizer.grid_range, -1, dtype=int)
    for i in range(len(voxel_labels)):
        z, y, x = coordinates[i]
        voxel_mat[x, y, z] = voxel_labels[i]
    xy_proj = np.full(voxelizer.grid_range[:2], -1, dtype=int)

    for x in range(voxelizer.grid_range[0]):
        for y in range(voxelizer.grid_range[1]):
            slice_z = voxel_mat[x, y, :]
            slice_z = slice_z[slice_z != -1]
            if len(slice_z) > 0:
                counts = np.bincount(slice_z)
                xy_proj[x, y] = np.argmax(counts)

    plt.imshow(xy_proj, cmap="jet", origin="lower")
    plt.colorbar()
    plt.savefig("voxels.png")
    return {"voxels": voxels, "voxel_labels": voxel_labels}
    # torch.save(
    #     {
    #         # "voxels": voxels,
    #         # "coordinates": coordinates,
    #         # "num_points_per_voxel": num_points_per_voxel,
    #         # "pc_voxel_id": pc_voxel_id,
    #         "voxel_labels": voxel_labels,
    #     },
    #     voxel_file,
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PCD files and voxelize them.")
    parser.add_argument(
        "--directory",
        type=str,
        default="/share/SemanticOPV2V/4LidarCampos",
        help="Directory containing the PCD files.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="semantic_campos.pcd",
        help="Source PCD file pattern to look for.",
    )
    parser.add_argument(
        "--vsize_xyz",
        type=float,
        nargs=3,
        default=[0.4, 0.4, 1],
        help="Voxel size in x, y, z dimensions.",
    )
    parser.add_argument(
        "--coors_range_xyz",
        type=float,
        nargs=6,
        default=[-70.4, -40, -3, 70.4, 40, 1],
        help="Coordinate range for voxelization.",
    )
    parser.add_argument(
        "--num_point_features",
        type=int,
        default=3,
        help="Number of point features.",
    )
    parser.add_argument(
        "--max_num_voxels",
        type=int,
        default=80000,
        help="Maximum number of voxels.",
    )
    parser.add_argument(
        "--max_num_points_per_voxel",
        type=int,
        default=5000,
        help="Maximum number of points per voxel.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Process device.",
    )
    parser.add_argument(
        "--cover",
        action="store_true",
        default=False,
        help="Overwrite existing voxel files.",
    )

    args = parser.parse_args()

    # pcd_files = find_files(args.directory, args.source)

    # pbar = tqdm(total=len(pcd_files), desc="Processing")

    # for pcd_file in pcd_files:
    #     process_pcd_file(
    #         pcd_file,
    #         args.source,
    #         args.vsize_xyz,
    #         args.coors_range_xyz,
    #         args.num_point_features,
    #         args.max_num_voxels,
    #         args.max_num_points_per_voxel,
    #         args.device,
    #         # args.cover,
    #     )
    #     pbar.set_description(f"Processing {pcd_file}")
    #     pbar.update(1)
    pcd_file = "/share/SemanticOPV2V-OpenMMLab/SemanticOPV2V/4LidarCampos/test/2021_08_18_19_48_05/1045/000068_semantic_campos.pcd"
    result = process_pcd_file(
        pcd_file,
        args.source,
        args.vsize_xyz,
        args.coors_range_xyz,
        args.num_point_features,
        args.max_num_voxels,
        args.max_num_points_per_voxel,
        args.device,
        # args.cover,
    )

# def depth2vox(depth_list, lidar_height=2.3):
#     """
#     Convert a list of depth maps to a 3D voxel representation.

#     Parameters:
#         depth_list (list of np.ndarray): List of depth maps.
#         lidar_height (float): Height offset for the lidar sensor (reference of coordinate system).

#     Returns:
#         np.ndarray: Voxel grid representing combined depth maps.
#     """

#     H, W = depth_list[0].shape[2], depth_list[0].shape[3]
#     k = np.array(
#         [
#             [W / (2.0 * tan(100 * pi / 360.0)), 0, W / 2.0],
#             [0, W / (2.0 * tan(100 * pi / 360.0)), H / 2.0],
#             [0, 0, 1],
#         ]
#     )

#     # 2D pixel coordinates
#     pixel_length = W * H
#     u_coord = np.tile(np.arange(W - 1, -1, -1), (H, 1)).reshape(pixel_length)
#     v_coord = np.tile(np.arange(H - 1, -1, -1)[:, None], (1, W)).reshape(pixel_length)

#     all_rotated_points = []

#     for i_img, depth in enumerate(depth_list):
#         depth = depth.detach().cpu().numpy()
#         depth = np.argmax(depth[0], axis=0).astype(np.float32) * 0.4

#         if i_img == 0:  # front
#             theta_z, translation = 0, np.array([2.5, 0, 1.0 - lidar_height])
#         elif i_img == 1:  # right
#             theta_z, translation = -100, np.array([0, -0.3, 1.8 - lidar_height])
#         elif i_img == 2:  # left
#             theta_z, translation = 100, np.array([0, 0.3, 1.8 - lidar_height])
#         elif i_img == 3:  # rear
#             theta_z, translation = 180, np.array([-2.2, 0, 1.5 - lidar_height])
#         else:
#             raise ValueError("Too many images: expected at most 4.")

#         # Rotation matrix for each orientation
#         theta_z_rad = math.radians(theta_z)
#         R_z = np.array(
#             [
#                 [np.cos(theta_z_rad), -np.sin(theta_z_rad), 0],
#                 [np.sin(theta_z_rad), np.cos(theta_z_rad), 0],
#                 [0, 0, 1],
#             ]
#         )

#         # Project depth to 3D points
#         p2d = np.array([u_coord, v_coord, np.ones_like(u_coord)])
#         p3d = np.dot(np.linalg.inv(k), p2d) * depth.flatten()
#         p3d = p3d.T
#         mask = p3d[:, 2] < 20  # Only keep points within 20m
#         p3d = p3d[mask]

#         # Transform points with rotation and translation
#         rotated_points = (R_z @ p3d.T).T + translation
#         all_rotated_points.extend(rotated_points)

#     # Convert all accumulated rotated points to voxel grid
#     vox = point2vox(np.array(all_rotated_points))
#     vox[vox > 0] = 1  # Ensure occupancy value consistency

#     return vox
