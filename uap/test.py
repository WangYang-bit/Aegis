import math
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest
import sys

# 根据需要调整模块导入路径
from opencood.utils import box_utils
import opencood.utils.common_utils as common_utils
from opencood.utils import pcd_utils
from opencood.utils.transformation_utils import x_to_world

from uap.models.world_model import WorldModel

def transfer():
    angle = 30 * math.pi / 180
    theta = np.array(
        [[math.cos(angle), math.sin(-angle), 5], [math.sin(angle), math.cos(angle), 6]]
    )
    # 变换1：可以实现缩放/旋转，这里为 [[1,0],[0,1]] 保存图片不变
    t1 = theta[:, [0, 1]]
    # 变换2：可以实现平移
    t2 = theta[:, [2]]
    pos = np.array([[5], [6]])
    npos = t1 @ pos + t2


def uniform_affine_quantization(x, qmin=0, qmax=255):
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - torch.round(min_val / scale)
    q = torch.round(x / scale + zero_point)
    x_quantized = scale * (q - zero_point)
    return zero_point, q, x_quantized


def asymmetric_quantization(x, qmin=0, qmax=255):
    min_val, max_val = x.min(), x.max()
    scale = (max_val - min_val) / (qmax - qmin)
    zero_point = qmin - min_val / scale
    q = torch.round((x - min_val) / scale)
    x_quantized = q * scale + min_val
    return q, x_quantized


def softmax_one_hot():
    # 定义softmax函数
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # 生成6个小于1的随机小数
    random_numbers = np.random.rand(6)

    # 计算【1，5，10，40】倍的softmax值
    multipliers = [float("-inf"), 5, 10, 40]
    softmax_values = {m: softmax(random_numbers * m) for m in multipliers}

    # 绘制折线图
    plt.figure(figsize=(10, 6))
    for m, values in softmax_values.items():
        plt.plot(values, label=f"{m}x")

    # 将x轴标签换为随机数的值
    plt.xticks(
        ticks=range(len(random_numbers)),
        labels=[f"{num:.2f}" for num in random_numbers],
    )
    plt.xlabel("Index")
    plt.ylabel("Softmax Value")
    plt.title("Softmax Values for Different Multipliers")
    plt.legend()
    plt.grid(True)
    plt.savefig("softmax_values.png")


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def test_scaled_dot_product_attention():
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

    # Q = torch.randn(10, 2112, 1024).to(device)
    # K = torch.randn(10, 2112, 1024).to(device)
    # V = torch.randn(10, 2112, 1024).to(device)

    Q = torch.randn(10, 8448, 256).to(device)
    K = torch.randn(10, 8448, 256).to(device)
    V = torch.randn(10, 8448, 256).to(device)

    _ = F.scaled_dot_product_attention(Q, K, V)  # 预热

    repeats = 5
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    start = time.time()
    for _ in range(repeats):
        _ = F.scaled_dot_product_attention(Q, K, V)
    end = time.time()

    avg_time = (end - start) / repeats
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024**2)
        print(f"平均执行时间: {avg_time:.6f} 秒, 峰值显存: {peak_mem:.2f} MB")
    else:
        print(f"平均执行时间: {avg_time:.6f} 秒")


def load_params_from_path(path):
    return torch.load(path, map_location="cpu")



def generate_box_corners(center, dims, yaw):
    """
    根据中心、尺寸和yaw生成box的8个角点
    参数:
        center: (2,) 或 (3,) 数组，表示box中心(x, y, z)
        dims: (l, w, h)
        yaw: 绕z轴旋转角度
    返回:
        corners: (8, 3) 数组，角点顺序按:
                 底面: [0,1,2,3] ；顶面: [4,5,6,7]
    """
    l, w, h = dims
    # 底面局部坐标 (顺时针或者逆时针均可，只要与原函数计算方式对应)
    bottom = np.array([
        [ l/2, -w/2],
        [ l/2,  w/2],
        [-l/2,  w/2],
        [-l/2, -w/2]
    ])
    R = np.array([[np.cos(yaw), -np.sin(yaw)],
                  [np.sin(yaw),  np.cos(yaw)]])
    bottom_rot = bottom @ R.T  # 旋转后的底面xy
    top_rot = bottom_rot.copy()
    bottom_xy = bottom_rot + np.array(center[:2])
    top_xy = top_rot + np.array(center[:2])
    bottom_z = center[2] - h/2
    top_z = center[2] + h/2

    corners = np.zeros((8, 3))
    corners[0:4, :2] = bottom_xy
    corners[0:4, 2] = bottom_z
    corners[4:8, :2] = top_xy
    corners[4:8, 2] = top_z
    return corners

class TestBoxUtils(unittest.TestCase):

    def setUp(self):
        # 构造一个简单的轴对齐立方体角点，其来自 create_bbx 函数定义：
        # 对称立方体, extent = [1, 1, 1]
        # 顺序为：
        # 0: [1, -1, -1]
        # 1: [1, 1, -1]
        # 2: [-1, 1, -1]
        # 3: [-1, -1, -1]
        # 4: [1, -1, 1]
        # 5: [1, 1, 1]
        # 6: [-1, 1, 1]
        # 7: [-1, -1, 1]
        self.corners_np = np.array([[1, -1, -1],
                                    [1,  1, -1],
                                    [-1,  1, -1],
                                    [-1, -1, -1],
                                    [1, -1,  1],
                                    [1,  1,  1],
                                    [-1,  1,  1],
                                    [-1, -1,  1]])
        # 增加 batch 维度: (1, 8, 3)
        self.corners_np = self.corners_np[np.newaxis, :]

        # 期望输出：
        # 根据函数实现，中心点为 (0,0,0)
        # l、w、h 均 = 2, yaw = 0 (order='lwh')
        self.expected_np = np.array([[0, 0, 0, 2, 2, 2, 0]])

        # 构造 torch 版本输入
        self.corners_torch = torch.from_numpy(self.corners_np).float()

    def test_corner_to_center_numpy(self):
        # 测试 numpy 版本的 corner_to_center
        output = box_utils.corner_to_center(self.corners_np, order='lwh')
        np.testing.assert_allclose(output, self.expected_np, rtol=1e-4)
    
    def test_corner_to_center_torch(self):
        # 测试 torch 版本的 corner_to_center_torch
        output = box_utils.corner_to_center_torch(self.corners_torch, order='lwh')
        np.testing.assert_allclose(output.cpu().detach().numpy(), self.expected_np, rtol=1e-4)
    
    def test_corner_to_center_torch_with_yaw(self):
        """
        测试角点非零旋转时，torch版的corner_to_center_torch是否能正确提取yaw
        """
        # 定义一个非轴对齐的box: center (2,3,4), dims: (4,2,2), yaw: 45° (pi/4)
        center = [2, 3, 4]
        dims = (4, 2, 2)
        yaw = math.pi / 4
        corners = generate_box_corners(center, dims, yaw)
        # 扩展 batch 维度: (1,8,3)
        corners = corners[np.newaxis, :]
        corners_torch = torch.from_numpy(corners).float()

        # 期望结果
        expected_center = np.array(center)
        expected_dims = np.array(dims)  # l, w, h
        expected = np.concatenate([expected_center,
                                   expected_dims, 
                                   np.array([yaw])], axis=0).reshape(1,7)
        
        output = box_utils.corner_to_center_torch(corners_torch, order='lwh')
        # 注意：yaw可能存在2pi的周期性问题，这里我们使用sin和cos的差值判断
        output_np = output.cpu().detach().numpy()
        np.testing.assert_allclose(output_np[0, :3], expected[0, :3], rtol=1e-4)  # center
        np.testing.assert_allclose(output_np[0, 3:6], expected[0, 3:6], rtol=1e-4)  # dims
        
        # 检测yaw: 由于yaw存在周期性，比较其sin和cos值
        self.assertAlmostEqual(np.cos(output_np[0, 6]), np.cos(expected[0, 6]), places=4)
        self.assertAlmostEqual(np.sin(output_np[0, 6]), np.sin(expected[0, 6]), places=4)
    
    def test_project_box3d_identity(self):
        # 测试 project_box3d：使用单位变换矩阵时，输出应与输入一致
        # 先利用 corner_to_center 函数构造一个 box3d (形状 (1,7))
        box3d = box_utils.corner_to_center(self.corners_np, order='lwh')
        # 将 box3d 转换为 8 个角点（利用 boxes_to_corners_3d 函数）
        corners3d = box_utils.boxes_to_corners_3d(box3d, order='lwh')
        # 单位矩阵
        T = np.eye(4)
        projected = box_utils.project_box3d(corners3d, T)
        # 输出与原始 corners3d 应该相等（误差在容许范围内）
        np.testing.assert_allclose(projected, corners3d, rtol=1e-4)
    
    def test_project_points_by_matrix_torch(self):
        # 测试 project_points_by_matrix_torch 使用单位矩阵
        points = torch.tensor([[1.0, 2.0, 3.0],
                               [4.0, 5.0, 6.0]])
        T = torch.eye(4)
        projected = box_utils.project_points_by_matrix_torch(points, T)
        np.testing.assert_allclose(projected.cpu().numpy(), points.cpu().numpy(), rtol=1e-4)

class TestPositionProject(unittest.TestCase):
    def test_identity(self):
        # 使用单位矩阵不做任何变换
        position = np.array([1, 2, 3])
        extrinsic = np.eye(4)
        projected = pcd_utils.position_project(position, extrinsic)
        self.assertTrue(np.allclose(projected, position))

    def test_translation(self):
        # 只做平移变换
        position = np.array([0, 0, 0])
        # 平移1,2,3
        extrinsic = np.array([[1, 0, 0, 1],
                              [0, 1, 0, 2],
                              [0, 0, 1, 3],
                              [0, 0, 0, 1]])
        projected = pcd_utils.position_project(position, extrinsic)
        expected = np.array([1, 2, 3])
        self.assertTrue(np.allclose(projected, expected))

    def test_rotation(self):
        # 绕 z 轴旋转 90 度
        position = np.array([1, 0, 0])
        theta = np.pi / 2
        extrinsic = np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                              [np.sin(theta),  np.cos(theta), 0, 0],
                              [0,              0,             1, 0],
                              [0,              0,             0, 1]])
        projected = pcd_utils.position_project(position, extrinsic)
        expected = np.array([0, 1, 0])
        self.assertTrue(np.allclose(projected, expected, atol=1e-6))
        
    def test_position_project(self):
        position = np.array([561.334228515625, -17.787120819091797, 0.0817679762840271])
        ego_pos = [561.9008178710938, -17.930795669555664, 1.957484245300293, 4.4824018478393555, -179.51243591308594, 1.986703872680664]
        world_to_ego = np.linalg.inv(x_to_world(ego_pos))
        projected = pcd_utils.position_project(position, world_to_ego)
        print(projected)
        self.assertTrue(np.allclose(projected, position))
if __name__ == '__main__':
    iou = np.zeros([])
    print(iou)