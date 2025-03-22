import math
import sys
from abc import abstractmethod
from collections import OrderedDict

import torch
import torch.nn.functional as F
from uap.config import uap_root
from torch.optim import Adam

sys.path.append(uap_root)
# print(sys.path)
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor
from opencood.utils import box_utils

# from mvp.perception.iou_util import oriented_box_intersection_2d


class PatchAttack(Adam):
    """An Attack Base Class"""

    def __init__(self, norm: str, cfg, device: torch.device, detector_attacker):
        """

        :param norm: str, [L0, L1, L2, L_infty]
        :param cfg:
        :param detector_attacker: this attacker should have attributes vlogger

        Args:
            loss_func ([torch.nn.Loss]): [a loss function to calculate the loss between the inputs and expeced outputs]
            norm (str, optional): [the attack norm and the choices are [L0, L1, L2, L_infty]]. Defaults to 'L_infty'.
            epsilons (float, optional): [the upper bound of perturbation]. Defaults to 0.05.
            max_iters (int, optional): [the maximum iteration number]. Defaults to 10.
            step_lr (float, optional): [the step size of attack]. Defaults to 0.01.
            device ([type], optional): ['cpu' or 'cuda']. Defaults to None.
        """
        lr = cfg["STEP_LR"]
        params = [detector_attacker.patch_obj.patch]
        super().__init__(params, lr=lr)

        self.cfg = cfg
        self.device = device
        self.norm = norm
        self.min_epsilon = 0.0
        self.max_epsilon = cfg["EPSILON"] / 255.0
        self.iter_step = cfg["ITER_STEP"]
        self.detector_attacker = detector_attacker

    def logger(self, detector, adv_tensor_batch, bboxes, loss_dict):
        vlogger = self.detector_attacker.vlogger
        # TODO: this is a manually appointed logger iter num 77(for INRIA Train)
        if vlogger:
            # print(loss_dict['loss'], loss_dict['det_loss'], loss_dict['tv_loss'])
            vlogger.note_loss(
                loss_dict["loss"], loss_dict["det_loss"], loss_dict["tv_loss"]
            )
            if vlogger.iter % 77 == 0:
                filter_box = self.detector_attacker.filter_bbox
                vlogger.write_tensor(
                    self.detector_attacker.universal_patch[0], "adv patch"
                )
                plotted = self.detector_attacker.plot_boxes(
                    adv_tensor_batch[0], filter_box(bboxes[0])
                )
                vlogger.write_cv2(plotted, f"{detector.name}")

    def patch_train(self, batch_data, model, attack_dict):
        losses = []
        output_dict = OrderedDict()
        output_dict = {}
        best_loss = 0xFFFFFFFF
        batch_data["bbox_tensor"] = attack_dict["object_bbox"].type(torch.float32)
        if self.cfg["regularization"]:
            with torch.no_grad():
                output_dict["ego_wo_adv"] = model.fusion_decoder(batch_data)
        # print(batch_data['bbox_tensor'])
        for iter in range(self.iter_step):
            adv_tensor_batch = self.patch_apply(batch_data, attack_dict)
            output_dict["ego"] = model.fusion_decoder(adv_tensor_batch)
            model.zero_grad()
            if self.cfg["regularization"]:
                if attack_dict["attack_tagget"] == "remove":
                    loss = self.remove_loss_regular(batch_data, output_dict)
                elif attack_dict["attack_tagget"] == "spoof":
                    loss = self.spoof_loss_regular(batch_data, output_dict)
            else:
                if attack_dict["attack_tagget"] == "remove":
                    loss = self.remove_loss(batch_data, output_dict)
                elif attack_dict["attack_tagget"] == "spoof":
                    loss = self.spoof_loss(batch_data, output_dict)

            # if loss.item() < best_loss:
            #     with torch.no_grad():
            #         data_dict = {'ego': batch_data}
            #         pred_box_tensor, pred_score = \
            #             model.post_processor.post_process(data_dict, output_dict)

            #     best_loss = loss.item()
            #     best_pred_bboxes = pred_box_tensor
            #     best_pred_scores = pred_score

            # print(loss)
            loss.backward(retain_graph=False)
            # print(self.detector_attacker.patch_obj.patch.grad)
            losses.append(float(loss))
            # update patch. for optimizer, using optimizer.step(). for PGD or others, using clamp and SGD.
            self.patch_update()
        # print(adv_tensor_batch, bboxes, loss_dict)
        # update training statistics on tensorboard
        # self.logger(model, adv_tensor_batch, bboxes, loss_dict)
        return torch.tensor(losses).mean()

    def attack(self, batch_data, model, attack_dict, apply_attack=True):
        output_dict = OrderedDict()
        output_dict = {}

        if apply_attack:
            batch_data["bbox_tensor"] = attack_dict["object_bbox"].type(torch.float32)
            adv_tensor_batch = self.patch_apply(batch_data, attack_dict)
        else:
            adv_tensor_batch = batch_data
        output_dict["ego"] = model.fusion_decoder(adv_tensor_batch)
        with torch.no_grad():
            data_dict = {"ego": batch_data}
            pred_box_tensor, pred_score = model.post_processor.post_process(
                data_dict, output_dict
            )
        result = {
            "pred_box_tensor": pred_box_tensor,
            "pred_score": pred_score,
            "adv_feature": adv_tensor_batch["spatial_features_2d"],
            "fused_feature": output_dict["ego"]["fused_feature"],
            "rm": output_dict["ego"]["rm"],
            "psm": output_dict["ego"]["psm"]
        }
        return result

    def attack_single_car(self, batch_data, model, attack_dict, apply_attack=True):
        output_dict = OrderedDict()
        output_dict = {}

        if apply_attack:
            batch_data["bbox_tensor"] = attack_dict["object_bbox"].type(torch.float32)
            adv_tensor_batch = self.patch_apply(batch_data, attack_dict)
        else:
            adv_tensor_batch = batch_data
        output_dict = model.fusion_decoder(adv_tensor_batch)
        pred_box_tensor, pred_score = model.post_processor.post_process_single_car(
            adv_tensor_batch, output_dict
        )
        
        result = {}
        num = len(pred_box_tensor)
        for i in range(num):
            result[batch_data['cav_list'][i]] = {
                "pred_box_tensor": pred_box_tensor[i],
                "pred_score": pred_score[i],
                "rm": output_dict["rm"][i],
                "psm": output_dict["psm"][i],
            }
        return result

    @abstractmethod
    def patch_update(self, **kwargs):
        self.step()
        self.zero_grad()

    @property
    def patch_obj(self):
        return self.detector_attacker.universal_patch
    
    def clamp_(self, min, max):
        self.detector_attacker.patch_obj.clamp_(min, max)


    def remove_loss(self, data_dict, output_dict):
        anchor_box = data_dict["anchor_box"]
        prob = F.sigmoid(output_dict["ego"]["psm"].permute(0, 2, 3, 1)).reshape(-1)
        proposals = VoxelPostprocessor.delta_to_boxes3d(
            output_dict["ego"]["rm"], anchor_box
        )[0]
        bbox_tensor = data_dict["bbox_tensor"]
        if bbox_tensor is not None:
            iou = torch.clip(
                self.iou_torch(
                    proposals[:, [0, 1, 2, 5, 4, 3, 6]],
                    bbox_tensor.tile((proposals.shape[0], 1)),
                ),
                min=0,
                max=1,
            )
        else:
            iou = (
                torch.ones(proposals.shape[0], dtype=torch.bool)
                .to(self.device)
                .detach()
            )
        box_mask = iou >= 0.01
        loss = (-1 * iou[box_mask] * torch.log(1 - prob[box_mask])).sum()
        return loss

    def remove_loss_regular(self, data_dict, output_dict):
        anchor_box = data_dict["anchor_box"]

        # Get proposals for both ego and ego_wo_adv
        proposals_adv = VoxelPostprocessor.delta_to_boxes3d(
            output_dict["ego"]["rm"], anchor_box
        )[0]
        proposals_wo_adv = VoxelPostprocessor.delta_to_boxes3d(
            output_dict["ego_wo_adv"]["rm"], anchor_box
        )[0]
        bbox_tensor = data_dict["bbox_tensor"]

        if bbox_tensor is not None:
            iou_adv = torch.clip(
                self.iou_torch(
                    proposals_adv[:, [0, 1, 2, 5, 4, 3, 6]],
                    bbox_tensor.tile((proposals_adv.shape[0], 1)),
                ),
                min=0,
                max=1,
            )

            iou_wo_adv = torch.clip(
                self.iou_torch(
                    proposals_wo_adv[:, [0, 1, 2, 5, 4, 3, 6]],
                    bbox_tensor.tile((proposals_wo_adv.shape[0], 1)),
                ),
                min=0,
                max=1,
            )

            iou = torch.clip(
                self.iou_torch(
                    proposals_wo_adv[:, [0, 1, 2, 5, 4, 3, 6]],
                    proposals_adv[:, [0, 1, 2, 5, 4, 3, 6]],
                ),
                min=0,
                max=1,
            )

        else:
            iou_adv = (
                torch.ones(proposals_adv.shape[0], dtype=torch.bool)
                .to(self.device)
                .detach()
            )
            iou_wo_adv = (
                torch.ones(proposals_wo_adv.shape[0], dtype=torch.bool)
                .to(self.device)
                .detach()
            )

        # Mask for boxes not overlapping with bbox_tensor
        box_mask_adv = iou_adv < 0.01
        box_mask_wo_adv = iou_wo_adv < 0.01
        box_mask = iou_adv >= 0.01

        # Calculate consistency loss
        prob_adv = F.sigmoid(output_dict["ego"]["psm"].permute(0, 2, 3, 1)).reshape(-1)
        prob_wo_adv = F.sigmoid(
            output_dict["ego_wo_adv"]["psm"].permute(0, 2, 3, 1)
        ).reshape(-1)

        attack_loss = (
            -1 * iou_adv[box_mask] * torch.log(1 - prob_adv[box_mask])
        ).mean()
        # consistency_loss = ((1 - iou[box_mask_adv]) * (prob_wo_adv[box_mask_adv] - prob_adv[box_mask_adv])**2).mean()
        # consistency_loss = (1 - iou[box_mask_adv]).mean() + torch.nn.functional.l1_loss(prob_wo_adv[box_mask_adv],prob_adv[box_mask_adv])
        consistency_loss = torch.nn.functional.l1_loss(
            prob_wo_adv[box_mask_adv], prob_adv[box_mask_adv]
        )
        loss = attack_loss + self.cfg["regular_parm"] * consistency_loss
        return loss

    def spoof_loss(self, data_dict, output_dict):
        anchor_box = data_dict["anchor_box"]
        prob = F.sigmoid(output_dict["ego"]["psm"].permute(0, 2, 3, 1)).reshape(-1)
        proposals = VoxelPostprocessor.delta_to_boxes3d(
            output_dict["ego"]["rm"], anchor_box
        )[0]
        bbox_tensor = data_dict["bbox_tensor"]
        if bbox_tensor is not None:
            iou = torch.clip(
                self.iou_torch(
                    proposals[:, [0, 1, 2, 5, 4, 3, 6]],
                    bbox_tensor.tile((proposals.shape[0], 1)),
                ),
                min=0,
                max=1,
            )
        else:
            iou = (
                torch.ones(proposals.shape[0], dtype=torch.bool)
                .to(self.device)
                .detach()
            )
        box_mask = iou >= 0.01
        epsilon = 1e-6
        loss = (1 * iou[box_mask] * torch.log(1 - prob[box_mask] + epsilon)).sum()
        return loss
    
    def spoof_loss_regular(self, data_dict, output_dict):
        anchor_box = data_dict["anchor_box"]

        # Get proposals for both ego and ego_wo_adv
        proposals_adv = VoxelPostprocessor.delta_to_boxes3d(
            output_dict["ego"]["rm"], anchor_box
        )[0]
        proposals_wo_adv = VoxelPostprocessor.delta_to_boxes3d(
            output_dict["ego_wo_adv"]["rm"], anchor_box
        )[0]
        bbox_tensor = data_dict["bbox_tensor"]

        if bbox_tensor is not None:
            iou_adv = torch.clip(
                self.iou_torch(
                    proposals_adv[:, [0, 1, 2, 5, 4, 3, 6]],
                    bbox_tensor.tile((proposals_adv.shape[0], 1)),
                ),
                min=0,
                max=1,
            )

            iou_wo_adv = torch.clip(
                self.iou_torch(
                    proposals_wo_adv[:, [0, 1, 2, 5, 4, 3, 6]],
                    bbox_tensor.tile((proposals_wo_adv.shape[0], 1)),
                ),
                min=0,
                max=1,
            )

            iou = torch.clip(
                self.iou_torch(
                    proposals_wo_adv[:, [0, 1, 2, 5, 4, 3, 6]],
                    proposals_adv[:, [0, 1, 2, 5, 4, 3, 6]],
                ),
                min=0,
                max=1,
            )

        else:
            iou_adv = (
                torch.ones(proposals_adv.shape[0], dtype=torch.bool)
                .to(self.device)
                .detach()
            )
            iou_wo_adv = (
                torch.ones(proposals_wo_adv.shape[0], dtype=torch.bool)
                .to(self.device)
                .detach()
            )

        # Mask for boxes not overlapping with bbox_tensor
        box_mask_adv = iou_adv < 0.01
        box_mask_wo_adv = iou_wo_adv < 0.01
        box_mask = iou_adv >= 0.01

        # Calculate consistency loss
        prob_adv = F.sigmoid(output_dict["ego"]["psm"].permute(0, 2, 3, 1)).reshape(-1)
        prob_wo_adv = F.sigmoid(
            output_dict["ego_wo_adv"]["psm"].permute(0, 2, 3, 1)
        ).reshape(-1)
        epsilon = 1e-6
        attack_loss = (1 * iou_adv[box_mask] * torch.log(1 - prob_adv[box_mask] + epsilon)).mean()
        # consistency_loss = ((1 - iou[box_mask_adv]) * (prob_wo_adv[box_mask_adv] - prob_adv[box_mask_adv])**2).mean()
        # consistency_loss = (1 - iou[box_mask_adv]).mean() + torch.nn.functional.l1_loss(prob_wo_adv[box_mask_adv],prob_adv[box_mask_adv])
        consistency_loss = torch.nn.functional.l1_loss(
            prob_wo_adv[box_mask_adv], prob_adv[box_mask_adv]
        )
        loss = attack_loss + self.cfg["regular_parm"] * consistency_loss
        return loss

    def patch_apply(self, data_dict, attack_dict):
        feature_index = attack_dict["obj_idx"]
        adv_tensor_batch = data_dict.copy()
        feature_map = torch.clone(data_dict["spatial_features_2d"])
        C, H, W = feature_map[0].size()
        patch_size = self.patch_obj.size()
        x, y = feature_index[0], feature_index[1]
        dx, dy = patch_size[1] / 2, patch_size[2] / 2

        x1, x2 = math.ceil(x - dx), math.ceil(x + dx)
        y1, y2 = math.ceil(y - dy), math.ceil(y + dy)

        x1_clip, x2_clip = max(x1, 0), min(x2, W)
        y1_clip, y2_clip = max(y1, 0), min(y2, H)
        pad_x1 = max(0, -x1)
        pad_y1 = max(0, -y1)

        with torch.no_grad():
            max_value = feature_map.max()
            min_value = feature_map.min()
        self.clamp_(min=-self.max_epsilon, max=self.max_epsilon)
        feature_map[0, :, y1_clip:y2_clip, x1_clip:x2_clip] = self.patch_obj[
            :,
            pad_y1 : pad_y1 + (y2_clip - y1_clip),
            pad_x1 : pad_x1 + (x2_clip - x1_clip),
        ]
        adv_tensor_batch["spatial_features_2d"]  = feature_map
        # print('feature device:',feature_map.device)
        return adv_tensor_batch

    def iou_torch(self, bboxes_a, bboxes_b):
        corners2d_a = torch.unsqueeze(
            box_utils.boxes_to_corners2d(bboxes_a, order="hwl")[:, :, :2], 0
        )
        corners2d_b = torch.unsqueeze(
            box_utils.boxes_to_corners2d(bboxes_b, order="hwl")[:, :, :2], 0
        )
        area_a = bboxes_a[:, 3] * bboxes_a[:, 4]
        area_b = bboxes_b[:, 3] * bboxes_b[:, 4]
        area_inter, _ = oriented_box_intersection_2d(corners2d_a, corners2d_b)
        area_inter = area_inter.squeeze()
        height_inter = torch.clip(
            torch.min(
                bboxes_a[:, 2] + 0.5 * bboxes_a[:, 5],
                bboxes_b[:, 2] + 0.5 * bboxes_b[:, 5],
            )
            - torch.max(
                bboxes_a[:, 2] - 0.5 * bboxes_a[:, 5],
                bboxes_b[:, 2] - 0.5 * bboxes_b[:, 5],
            ),
            min=0,
            max=5,
        )
        iou = (
            area_inter
            * height_inter
            / (
                area_a * bboxes_a[:, 5]
                + area_b * bboxes_b[:, 5]
                - area_inter * height_inter
            )
        )
        return iou

    def begin_attack(self):
        """
        to tell attackers: now, i'm begin attacking!
        """
        pass

    def end_attack(self):
        """
        to tell attackers: now, i'm stop attacking!
        """
        pass
