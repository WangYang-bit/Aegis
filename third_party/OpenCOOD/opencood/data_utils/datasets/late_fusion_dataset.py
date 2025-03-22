# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for late fusion
"""
import random
import math
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

import opencood.data_utils.datasets
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.datasets import basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils import box_utils
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class LateFusionDataset(basedataset.BaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    detection outputs to ego.
    """
    def __init__(self, params, visualize, train=True):
        super(LateFusionDataset, self).__init__(params, visualize, train)
        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = build_postprocessor(params['postprocess'], train)

    def __getitem__(self, idx):
        base_data_dict = self.retrieve_base_data(idx)
        if self.train:
            reformat_data_dict = self.get_item_train(base_data_dict)
        else:
            reformat_data_dict = self.get_item_test(base_data_dict)

        return reformat_data_dict

    def get_item_single_car(self, selected_cav_base):
        """
        Process a single CAV's information for the train/test pipeline.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}
        scenario_id = selected_cav_base['scenario_id']
        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        # lidar unprocessed
        selected_cav_processed.update({'lidar': lidar_np.copy()})

        lidar_np = shuffle_points(lidar_np)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # remove points that hit ego vehicle
        lidar_np = mask_ego_points(lidar_np)



        # generate the bounding box(n, 7) under the cav's space
        object_bbx_center, object_bbx_mask, object_ids = \
            self.post_processor.generate_object_center([selected_cav_base],
                                                       selected_cav_base[
                                                           'params'][
                                                           'lidar_pose'])
        order = self.post_processor.params.get('order', 'lwh').lower()
        vehicle_size = self.params.get('vehicle_size', [4.5, 2.0, 1.5])
        if order == 'hwl':
            # Convert vehicle size from [length, width, height] to [height, width, length]
            self.params['vehicle_size'] = [vehicle_size[2], vehicle_size[1], vehicle_size[0]]
        # compute ego bbox (7-dim: x, y, z, l, w, h, yaw) in ego coordinate
        lidar_pose = np.array(selected_cav_base['params']['lidar_pose'])
        true_ego_pos = np.array(selected_cav_base['params']['true_ego_pos'])
        # assume that both lidar_pose and true_ego_pos have at least 4 elements: x, y, z, yaw
        # transform the true ego position to the lidar coordinate frame
        translation = true_ego_pos[:3] - lidar_pose[:3]
        ego_yaw = lidar_pose[3] if len(lidar_pose) > 3 else 0
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        x_rel = translation[0] * cos_yaw - translation[1] * sin_yaw
        y_rel = translation[0] * sin_yaw + translation[1] * cos_yaw
        z_rel = translation[2]
        true_ego_yaw = true_ego_pos[3] if len(true_ego_pos) > 3 else 0
        rel_yaw = true_ego_yaw - ego_yaw
        # use default vehicle size (length, width, height) if not specified in params
        vehicle_size = self.params.get('vehicle_size', [4.5, 2.0, 1.5])
        ego_bbox_center = np.array([x_rel, y_rel, z_rel, vehicle_size[0], vehicle_size[1], vehicle_size[2], rel_yaw])
        selected_cav_processed.update({'ego_bbox': ego_bbox_center})

        # data augmentation
        lidar_np, object_bbx_center, object_bbx_mask = \
            self.augment(lidar_np, object_bbx_center, object_bbx_mask)
        
        

        if self.visualize:
            selected_cav_processed.update({'origin_lidar': lidar_np})

        selected_cav_processed.update({'lidar_pose':
                                        selected_cav_base['params']['lidar_pose']})
        # pre-process the lidar to voxel/bev/downsampled lidar
        lidar_dict = self.pre_processor.preprocess(lidar_np)
        selected_cav_processed.update({'processed_lidar': lidar_dict})

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()
        selected_cav_processed.update({'anchor_box': anchor_box})

        selected_cav_processed.update({'object_bbx_center': object_bbx_center,
                                       'object_bbx_mask': object_bbx_mask,
                                       'object_ids': object_ids})

        # generate targets label
        label_dict = \
            self.post_processor.generate_label(
                gt_box_center=object_bbx_center,
                anchors=anchor_box,
                mask=object_bbx_mask)
        selected_cav_processed.update({'label_dict': label_dict})
        selected_cav_processed.update({'scenario_id': scenario_id})
        return selected_cav_processed

    def get_item_train(self, base_data_dict):
        processed_data_dict = OrderedDict()

        # during training, we return a random cav's data
        if not self.visualize:
            selected_cav_id, selected_cav_base = \
                random.choice(list(base_data_dict.items()))
        else:
            selected_cav_id, selected_cav_base = \
                list(base_data_dict.items())[0]

        selected_cav_processed = self.get_item_single_car(selected_cav_base)
        processed_data_dict.update({'ego': selected_cav_processed})

        return processed_data_dict

    def get_item_test(self, base_data_dict):
        processed_data_dict = OrderedDict()
        ego_id = -1
        ego_lidar_pose = []

        # first find the ego vehicle's lidar pose
        for cav_id, cav_content in base_data_dict.items():
            if cav_content['ego']:
                ego_id = cav_id
                ego_lidar_pose = cav_content['params']['lidar_pose']
                break

        assert ego_id != -1
        assert len(ego_lidar_pose) > 0

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in base_data_dict.items():
            distance = \
                math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                           ego_lidar_pose[0])**2 + (
                                      selected_cav_base['params'][
                                          'lidar_pose'][1] - ego_lidar_pose[
                                          1])**2)
            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            # find the transformation matrix from current cav to ego.
            cav_lidar_pose = selected_cav_base['params']['lidar_pose']
            transformation_matrix = x1_to_x2(cav_lidar_pose, ego_lidar_pose)

            selected_cav_processed = \
                self.get_item_single_car(selected_cav_base)
            
            selected_cav_processed.update({'transformation_matrix':
                                               transformation_matrix})
            # update_cav = "ego" if cav_id == ego_id else cav_id
            # processed_data_dict.update({update_cav: selected_cav_processed})
            processed_data_dict.update({cav_id: selected_cav_processed})

        return processed_data_dict

    def collate_batch_test(self, batch):
        """
        Customized collate function for pytorch dataloader during testing
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # currently, we only support batch size of 1 during testing
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        batch = batch[0]

        output_dict = {}
        total_object_bbox = {}
        # for late fusion, we also need to stack the lidar for better
        # visualization
        if self.visualize:
            projected_lidar_list = []
            origin_lidar = []

        for cav_id, cav_content in batch.items():
            output_dict.update({cav_id: {}})
            # shape: (1, max_num, 7)
            object_bbx_center = \
                torch.from_numpy(np.array([cav_content['object_bbx_center']]))
            object_bbx_mask = \
                torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
            object_ids = cav_content['object_ids']
            gt_bboxes = []
            for idx, object_id in enumerate(object_ids):
                if object_id not in total_object_bbox:
                    total_object_bbox.update({object_id: object_bbx_center[0][idx]})
                gt_bboxes.append(object_bbx_center[0][idx])
            gt_bboxes_list = [bbox.cpu().numpy() if isinstance(bbox, torch.Tensor) else np.array(bbox) for bbox in gt_bboxes]
            gt_bboxes_array = np.stack(gt_bboxes_list) if len(gt_bboxes_list) > 0 else np.array([])
            # the anchor box is the same for all bounding boxes usually, thus
            # we don't need the batch dimension.
            if cav_content['anchor_box'] is not None:
                output_dict[cav_id].update({'anchor_box':
                    torch.from_numpy(np.array(
                        cav_content[
                            'anchor_box']))})
            if self.visualize:
                transformation_matrix = cav_content['transformation_matrix']
                origin_lidar = [cav_content['origin_lidar']]

                projected_lidar = cav_content['origin_lidar']
                projected_lidar[:, :3] = \
                    box_utils.project_points_by_matrix_torch(
                        projected_lidar[:, :3],
                        transformation_matrix)
                projected_lidar_list.append(projected_lidar)

            ego_lidar_pose = cav_content['lidar_pose']
            # processed lidar dictionary
            processed_lidar_torch_dict = \
                self.pre_processor.collate_batch(
                    [cav_content['processed_lidar']])
            # label dictionary
            label_torch_dict = \
                self.post_processor.collate_batch([cav_content['label_dict']])

            # save the transformation matrix (4, 4) to ego vehicle
            transformation_matrix_torch = \
                torch.from_numpy(
                    np.array(cav_content['transformation_matrix'])).float()

            output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                        'object_bbx_mask': object_bbx_mask,
                                        'processed_lidar': processed_lidar_torch_dict,
                                        'label_dict': label_torch_dict,
                                        'object_ids': object_ids,
                                        'transformation_matrix': transformation_matrix_torch,
                                        'scenario_id': cav_content['scenario_id'],
                                        'lidar':torch.from_numpy(cav_content['lidar']),
                                        'lidar_pose': ego_lidar_pose,
                                        'ego_bbox': cav_content['ego_bbox'],
                                        'gt_bboxes': gt_bboxes_array,
                                        })

            if self.visualize:
                origin_lidar = \
                    np.array(
                        downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict[cav_id].update({'origin_lidar': origin_lidar})

        # for cav_id, cav_content in output_dict.items():
        #     if int(cav_id) in total_object_bbox:
        #         cav_content.update({'ego_bbox': total_object_bbox[int(cav_id)]})
        
        # gt_bboxes_array = np.stack([bbox.cpu().numpy() if isinstance(bbox, torch.Tensor) else np.array(bbox) for bbox in list(total_object_bbox.values())])
        # for cav_id, cav_content in output_dict.items():
        #     cav_content.update({'gt_bboxes': gt_bboxes_array})

        if self.visualize:
            projected_lidar_stack = torch.from_numpy(
                np.vstack(projected_lidar_list))
            output_dict['ego'].update({'origin_lidar': projected_lidar_stack})
            output_dict['ego'].update({'lidar_pose': ego_lidar_pose})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)
        gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

        return pred_box_tensor, pred_score, gt_box_tensor
