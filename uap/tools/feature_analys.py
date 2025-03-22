import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from uap.utils.box_utils import box_within_lidar, boxes_overlap, get_2d_polygon


def similarity_analys(features):
    num_tensors = len(features)
    similarity_matrix = torch.zeros(num_tensors, num_tensors)

    for i in range(num_tensors):
        for j in range(num_tensors):
            # flat feature
            tensor_i_flat = features[i].contiguous().view(-1)
            tensor_j_flat = features[j].contiguous().view(-1)
            # caculate cosin similarity
            similarity = F.cosine_similarity(tensor_i_flat, tensor_j_flat, dim=0)

            similarity_matrix[i, j] = similarity

    return similarity_matrix


def get_feature_index(obj_bbox, dataset):
    if isinstance(obj_bbox, torch.Tensor):
        bbox = obj_bbox.cpu().numpy()
    else:
        bbox = obj_bbox
    lidar_range = dataset.pre_processor.params["cav_lidar_range"]
    voxel_size = dataset.pre_processor.params["args"]["voxel_size"]
    feature_index = ((np.floor(bbox[:3] - lidar_range[:3]) / voxel_size) / 2).astype(
        np.int32
    )
    return feature_index


def get_random_index(data_dict, dataset, num=1, mode="remove"):
    result = {}
    if mode == "remove":
        num_to_select = min(num, len(data_dict["object_ids"]))
        selected_obj_ids = random.sample(data_dict["object_ids"], num_to_select)
        for obj_id in selected_obj_ids:
            object_index = data_dict["object_ids"].index(obj_id)
            remove_bbox = data_dict["object_bbx_center"][0][object_index]
            feature_index = get_feature_index(remove_bbox, dataset)
            result[obj_id] = (feature_index, remove_bbox)
        return result

    elif mode == "spoof":
        selected_obj_ids = [data_dict["object_ids"][-1] + i for i in range(num)]
        obj_bbox = data_dict["object_bbx_center"][0]
        obj_bbox_mask = data_dict["object_bbx_mask"][0] == 1
        obj_bbox = obj_bbox[obj_bbox_mask]
        boxes = obj_bbox.cpu().numpy()

        for obj_id in selected_obj_ids:
            lidar_range = dataset.pre_processor.params["cav_lidar_range"]
            spoof_box = generate_new_box(boxes, lidar_range)
            if spoof_box is None:
                return result
            feature_index = get_feature_index(spoof_box, dataset)
            result[obj_id] = (feature_index, spoof_box)
        return result

    else:
        print("unrecongnize mode!")
        return None


def generate_new_box(boxes, lidar_range, max_trials=50000):
    candidate_dims = np.mean(boxes[:, 3:6], axis=0)  # [l, h, w]
    candidate_cz = np.mean(boxes[:, 2])
    l_dim, h_dim, w_dim = candidate_dims



    boxes_list = boxes.tolist()
    if len(boxes_list) == 0:
        return None
    idx = 0
    while True:
        idx = random.randint(0, len(boxes)-1)
        chosen_box = boxes[idx]
        cx, cy, _, chosen_l, chosen_h, chosen_w, chosen_yaw = chosen_box
        for offset_distance in range(10, 100):
            new_cx = cx + offset_distance * np.cos(chosen_yaw)
            new_cy = cy + offset_distance * np.sin(chosen_yaw)
            candidate_box = [new_cx, new_cy, candidate_cz, l_dim, h_dim, w_dim, chosen_yaw]

            if not box_within_lidar(candidate_box, lidar_range):
                continue
            
            overlap = False
            for box in boxes_list:
                poly1 = get_2d_polygon(candidate_box)
                poly2 = get_2d_polygon(box)
                if poly1.intersection(poly2).area > 0:
                    overlap = True
                    break
            if not overlap:
                return torch.from_numpy(np.array(candidate_box))


def get_obj_feature(feature_map, feature_index, feature_size):
    C, H, W = feature_map[0].size()
    x, y = feature_index[0], feature_index[1]
    dx, dy = feature_size[0], feature_size[1]

    x1, x2 = x - dx, x + dx
    y1, y2 = y - dy, y + dy

    sub_feature_map = torch.zeros((C, 2 * dx, 2 * dy))

    x1_clip, x2_clip = max(x1, 0), min(x2, W)
    y1_clip, y2_clip = max(y1, 0), min(y2, H)

    pad_x1 = max(0, -x1)
    pad_y1 = max(0, -y1)

    sub_feature_map[
        :, pad_y1 : pad_y1 + (y2_clip - y1_clip), pad_x1 : pad_x1 + (x2_clip - x1_clip)
    ] = feature_map[0, :, y1_clip:y2_clip, x1_clip:x2_clip]

    return sub_feature_map


def feature_visualize(feature_map, save_file=None):
    feature_map = torch.max(feature_map, dim=0).values.detach().cpu().numpy()
    plt.imshow(feature_map, cmap="viridis", interpolation="nearest")
    if save_file is not None:
        plt.savefig(save_file)


if __name__ == "__main__":
    feature_map = torch.rand((64, 168, 98))

    feature_visualize(feature_map)
