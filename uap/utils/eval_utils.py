import os

import numpy as np
import yaml
from datetime import datetime

from opencood.utils import common_utils, box_utils
from opencood.hypes_yaml import yaml_utils
from opencood.utils.eval_utils import  calculate_ap


def setup_eval(hypes, file_name=""):
    """
    Create folder for saved model based on current timestep

    Parameters
    ----------
    hypes: dict
        Config yaml dictionary for evaluation:
    """
    current_time = datetime.now()

    folder_name = current_time.strftime("_%Y_%m_%d_%H_%M_%S")

    folder_name = file_name + folder_name

    current_path = os.path.dirname(__file__)
    current_path = os.path.join(current_path, '../logs')

    full_path = os.path.join(current_path, folder_name)

    if not os.path.exists(full_path):
        if not os.path.exists(full_path):
            try:
                os.makedirs(full_path)
            except FileExistsError:
                pass
        # save the yaml file
        save_name = os.path.join(full_path, 'config.yaml')
        with open(save_name, 'w') as outfile:
            yaml.dump(hypes, outfile)

    return full_path

def eval_attack(attack_boxes, det_boxes, attack_dict, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    attack_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_boxes : torch.Tensor
        The clean bounding box.
    iou_thresh : float
        The iou thresh.
    """
    obj_bbox = attack_dict['object_bbox'].unsqueeze(0)
    obj_bbox_corner = box_utils.boxes_to_corners_3d(obj_bbox,'hwl')
    obj_bbox = common_utils.torch_tensor_to_numpy(obj_bbox_corner)
    obj_polygon = common_utils.convert_format(obj_bbox)[0]
    iou_attack = np.array([])
    ious_det = np.array([0])
    if attack_boxes is not None:
        # convert bounding boxes to numpy array
        attack_boxes = common_utils.torch_tensor_to_numpy(attack_boxes)
        attack_polygon_list = list(common_utils.convert_format(attack_boxes))
        iou_attack = common_utils.compute_iou(obj_polygon , attack_polygon_list)
    if det_boxes is not None:
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        ious_det = common_utils.compute_iou(obj_polygon , det_polygon_list)
        
    if attack_dict['attack_tagget'] == 'remove':
        if iou_attack.size == 0:
            return True
        return np.max(ious_det) > iou_thresh and np.max(iou_attack) < iou_thresh
    elif attack_dict['attack_tagget'] == 'spoof':
        if iou_attack.size == 0:
            return False
        return np.max(ious_det) < iou_thresh and np.max(iou_attack) > iou_thresh
    else:
        raise NotImplementedError()

def caluclate_tp_fp_expt_attack(det_boxes, det_score, gt_boxes, result_stat, attack_dict, iou_thresh):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    gt = gt_boxes.shape[0]

    obj_bbox = attack_dict['object_bbox'].unsqueeze(0)
    obj_bbox_corner = box_utils.boxes_to_corners_3d(obj_bbox,'hwl')
    obj_bbox = common_utils.torch_tensor_to_numpy(obj_bbox_corner)
    
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)


        det_polygon_list = list(common_utils.convert_format(det_boxes))
        gt_polygon_list = list(common_utils.convert_format(gt_boxes))
        obj_polygon = common_utils.convert_format(obj_bbox)[0]

        ious_gt = common_utils.compute_iou(obj_polygon, gt_polygon_list)
        ious_det = common_utils.compute_iou(obj_polygon , det_polygon_list)
        # Remove gt polygons with iou greater than iou_thresh with obj_polygon
        gt_polygon_list = [gt_polygon for i, gt_polygon in enumerate(gt_polygon_list) if ious_gt[i] <= iou_thresh]
        
        # Remove det polygons with iou greater than iou_thresh with obj_polygon
        det_polygon_list = [det_polygon for i, det_polygon in enumerate(det_polygon_list) if ious_det[i] <= iou_thresh]
        # Remove det scores with iou greater than iou_thresh with obj_polygon
        det_score = np.array([det_score[i] for i in range(len(det_score)) if ious_det[i] <= iou_thresh])
        
        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score)
        det_score = det_score[score_order_descend] # from high to low

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)

        result_stat[iou_thresh]['score'] += det_score.tolist()

    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def eval_final_result(detector_name, attack_result, clean_result, save_path):
    """
    Calculate the attack success rate.

    Parameters
    ----------
    result_stat : dict
        The attack results.
    save_file : str
        The save file path.
    """
    dump_dict = {}
    attack_success = attack_result['attack_success']
    attack_success_num = sum(attack_success)

    attack_ap_30, attack_mrec_30, attack_mpre_30 = calculate_ap(attack_result, 0.30, False)
    attack_ap_50, attack_mrec_50, attack_mpre_50 = calculate_ap(attack_result, 0.50, False)
    attack_ap_70, attack_mrec_70, attack_mpre_70 = calculate_ap(attack_result, 0.70, False)

    clean_ap_30, clean_mrec_30, clean_mpre_30 = calculate_ap(clean_result, 0.30, False)
    clean_ap_50, clean_mrec_50, clean_mpre_50 = calculate_ap(clean_result, 0.50, False)
    clean_ap_70, clean_mrec_70, clean_mpre_70 = calculate_ap(clean_result, 0.70, False)

    print('eval results for %s: ' % detector_name)
    print('attack success num: ', attack_success_num)
    print('attack success rate: ', attack_success_num/len(attack_success))
    print('ap_30_diff_rate:', (attack_ap_30 - clean_ap_30)/clean_ap_30)
    print('ap_50_diff_rate:', (attack_ap_50 - clean_ap_50)/clean_ap_50)
    print('ap_70_diff_rate:', (attack_ap_70 - clean_ap_70)/clean_ap_70)

    dump_dict.update({'attack_success_num': attack_success_num,
                      'attack_success_rate': attack_success_num/len(attack_success),
                      'attack_failed_scene': attack_result['attack_failed_scene'],
                      'attack_ap_30': attack_ap_30,
                      'attack_ap_50': attack_ap_50,
                      'attack_ap_70': attack_ap_70,
                      'clean_ap_30': clean_ap_30,
                      'clean_ap_50': clean_ap_50,
                      'clean_ap_70': clean_ap_70,
                      'ap_30_diff_rate': (attack_ap_30 - clean_ap_30)/clean_ap_30,
                      'ap_50_diff_rate': (attack_ap_50 - clean_ap_50)/clean_ap_50,
                      'ap_70_diff_rate': (attack_ap_70 - clean_ap_70)/clean_ap_70,})
    
    output_file = '%s_eval.yaml' % detector_name 
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, output_file))



