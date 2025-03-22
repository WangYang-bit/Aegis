import numpy as np
from opencood.utils import box_utils, common_utils
from opencood.utils.common_utils import convert_format, compute_iou, torch_tensor_to_numpy
import torch
import numpy as np
from shapely.geometry import Polygon

def compute_iou(box, boxes):
    """
    Compute the IoU between a box and a list of boxes.
    Args:
        box: A tensor of shape (8,3) representing a box in the format of [x1, y1, x2, y2].
        boxes: A tensor of shape (N, 8,3) representing a list of boxes in the format of [x1, y1, x2, y2].
    Returns:
        ious: A tensor of shape (N,) representing the IoU between the box and each box in the list
    """

    box_polygon = common_utils.torch_tensor_to_numpy(box.unsqueeze(0))
    if boxes is None or len(boxes) == 0:
        return np.zeros(1)
    if isinstance(boxes, torch.Tensor):
        boxes_polygon = common_utils.torch_tensor_to_numpy(boxes)
    elif isinstance(boxes, list):
        # 假设 boxes 为 (8,3) tensor 列表
        boxes_numpy_list = [common_utils.torch_tensor_to_numpy(b) for b in boxes]
        boxes_polygon = np.stack(boxes_numpy_list)
    else:
        raise TypeError("boxes must be tensor or (8,3) tensor list")
    
    box_polygon= common_utils.convert_format(box_polygon)[0]
    boxes_polygon_list = list(common_utils.convert_format(boxes_polygon))
    ious = common_utils.compute_iou(box_polygon, boxes_polygon_list)
    return ious

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: a (N,8,3) tensor of boxes
    :return: a matrix of overlaps [boxes1 count, boxes2 count]
    """
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    if boxes1 is None or boxes2 is None:
        return np.zeros(0)
    boxes1_numpy = torch_tensor_to_numpy(boxes1)
    boxes2_numpy = torch_tensor_to_numpy(boxes2)

    boxes1 = convert_format(boxes1_numpy)
    boxes2 = convert_format(boxes2_numpy)
    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for i in range(overlaps.shape[0]):
        box1 = boxes1[i]
        overlaps[i, :] = common_utils.compute_iou(box1, boxes2)
    return overlaps

def check_overlap(center_box, boxes, order):
    """
    Determine which boxes in 'boxes' overlap with the given center_box using simple AABB testing.
    
    Parameters:
        center_box (Tensor): A tensor of shape (8, 3) representing a single box.
        boxes (Tensor): A tensor of shape (N, 8, 3) representing multiple boxes.
        order (str): The corner order required by the detector (e.g., "xyz").
        
    Returns:
        Tensor: A boolean tensor of shape (N,); True indicates an overlap with center_box.
    """
    # Expand center_box to shape (1, 7) and compute its 8 corners.
    # center_box_exp = center_box.unsqueeze(0)  # (1,7)
    # center_corners = box_utils.boxes_to_corners_3d(center_box_exp, order=order)[0]  # (8,3)
    center_min = center_box.min(dim=0)[0]  # (3,)
    center_max = center_box.max(dim=0)[0]  # (3,)

    # Convert each box in 'boxes' to its 8 corners and compute the axis aligned bounding boxes (AABB).
    # boxes_corners = box_utils.boxes_to_corners_3d(boxes, order=order)  # (N,8,3)
    boxes_min = boxes.min(dim=1)[0]  # (N,3)
    boxes_max = boxes.max(dim=1)[0]  # (N,3)

    overlap = ((center_min <= boxes_max) & (center_max >= boxes_min)).all(dim=1)
    return overlap


def get_2d_polygon(box):
    """
    box: [cx, cy, cz, l, h, w, yaw]
    返回二维旋转矩形，对应 (x,y) 投影，利用长 l 和宽 w
    """
    cx, cy, _, l, _, w, yaw = box
    half_l, half_w = l / 2, w / 2
    pts = np.array(
        [[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]]
    )
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    pts_rot = (R @ pts.T).T + np.array([cx, cy])
    return Polygon(pts_rot)


def boxes_overlap(box1, box2):
    """
    判断两个 box 是否重叠。
    在 x,y 投影为多边形交集，并检查 z 投影是否重叠（即 [cz, cz+h] 区间相交）。
    """
    poly1 = get_2d_polygon(box1)
    poly2 = get_2d_polygon(box2)
    if poly1.intersection(poly2).area > 0:
        z1, z2 = box1[2], box1[2] + box1[4]
        z3, z4 = box2[2], box2[2] + box2[4]
        if max(z1, z3) < min(z2, z4):
            return True
    return False


def box_within_lidar(box, lidar_range):
    """
    lidar_range: [min_x, min_y, min_z, max_x, max_y, max_z]
    检查 box 是否完全在 lidar 范围内（检查 x,y 投影的所有角点及 z 轴范围）。
    """
    min_x, min_y, min_z, max_x, max_y, max_z = lidar_range
    cx, cy, cz, l, h, w, yaw = box
    # if cz < min_z or (cz + h) > max_z:
    #     return False

    poly = get_2d_polygon(box)
    for x, y in np.array(poly.exterior.coords):
        if x < min_x or x > max_x or y < min_y or y > max_y:
            return False
    return True

