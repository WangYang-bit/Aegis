import os
from collections import OrderedDict

import opencood.hypes_yaml.yaml_utils as yaml_utils
import torch
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from torch.utils.data import DataLoader
from tqdm import tqdm

from uap.attack.patch_attack import PatchAttack
from uap.attack.target_perturb_attack import Target_Perturbation_Attack
from uap.config import data_root, model_root
from uap.patch import PatchManager
from uap.tools.feature_analys import *



class UniversalAttacker(object):
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device
        self.patch_obj = PatchManager(config["patch"], device)
        self.detectors = {}
        self.anchor_boxes = {}
        model_list = config["detectors"]
        for detector_name in model_list:
            model_dir = os.path.join(model_root, detector_name)
            config_file = os.path.join(model_dir, "config.yaml")
            hypes = yaml_utils.load_yaml(config_file, None)
            hypes["root_dir"] = os.path.join(data_root, "OPV2V/train")
            hypes["validate_dir"] = os.path.join(data_root, "OPV2V/test")
            detector = train_utils.create_model(hypes)
            detector.to(device)
            initial_epoch, detector = train_utils.load_saved_model(model_dir, detector)
            detector.eval()
            self.detectors[detector_name] = detector
            self.anchor_boxes[detector_name] = torch.from_numpy(
                detector.post_processor.generate_anchor_box()
            )
        self.init_attaker()

    def init_attaker(self):
        cfg = self.config["ATTACKER"]
        if cfg["METHOD"] == "target_perturbation":
            self.attacker = Target_Perturbation_Attack(
                norm="L_infty", device=self.device, cfg=cfg, detector_attacker=self
            )
        elif cfg["METHOD"] == "patch":
            self.attacker = PatchAttack(
                norm="L_infty", device=self.device, cfg=cfg, detector_attacker=self
            )
        else:
            raise ValueError("Unrecognized attack method!")

    def freeze_patch(self):
        self.patch_obj.freeze()

    @property
    def universal_patch(self):
        """This is for convenient calls.

        :return: the adversarial patch tensor.
        """
        return self.patch_obj.patch

    def attack_train(self, batch_data, attack_dict, mode="sequential"):
        """Call the base attack method to optimize the patch.

        :param batch_data: feature batch input.
        :param mode: attack mode(To define the updating behavior of multi-model ensembling.)
        :return: loss
        """
        detectors_loss = []
        self.attacker.begin_attack()
        if mode == "optim" or mode == "sequential":
            for detector_name, detector in self.detectors.items():
                with torch.no_grad():
                    batch_data["anchor_box"] = self.anchor_boxes[detector_name]
                    data_dict = detector.feature_encoder(batch_data)
                loss = self.attacker.patch_train(data_dict, detector, attack_dict)
                detectors_loss.append(loss)
        elif mode == "parallel":
            detectors_loss = self.parallel_attack(batch_data)
        self.attacker.end_attack()
        return torch.tensor(detectors_loss).mean()

    def attack(self, data_dict, detector, attack_dict, apply_attack=True):
        """Call the base attack method to optimize the patch.

        :param batch_data: feature batch input.

        :return: attack result
        """
        return self.attacker.attack(data_dict, detector, attack_dict, apply_attack)

    def attack_single_car(self, data_dict, detector, attack_dict, apply_attack=True):
        """Call the base attack method to optimize the patch.

        :param batch_data: feature batch input.

        :return: attack result
        """
        return self.attacker.attack_single_car(
            data_dict, detector, attack_dict, apply_attack
        )

    def parallel_attack(self, img_tensor_batch: torch.Tensor):
        """Multi-model ensembling: parallel attack mode.
            To average multi-updates to obtain the ultimate patch update in a single iter.
            FIXME: Not fully-supported currently.
        :param img_tensor_batch:
        :return: loss
        """
        detectors_loss = []
        patch_updates = torch.zeros(self.universal_patch.shape).to(self.device)
        for detector in self.detectors:
            patch_tmp, loss = self.attacker.non_targeted_attack(
                img_tensor_batch, detector
            )
            patch_update = patch_tmp - self.universal_patch
            patch_updates += patch_update
            detectors_loss.append(loss)
        self.patch_obj.update_(
            (self.universal_patch + patch_updates / len(self.detectors)).detach_()
        )
        return detectors_loss

    def get_detectors_num(self):
        return len(self.detectors)


if __name__ == "__main__":
    detectors = {}
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    result_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "result")
    os.makedirs(result_path, exist_ok=True)
    # model_dir = os.path.join(model_root, "pointpillar_attentive_fusion/config.yaml")
    # model_dir = os.path.join(model_root, "pointpillar_CoBEVT/config.yaml")
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")
    # model_dir = os.path.join(model_root, "pointpillar_v2vnet/config.yaml")
    # torch.autograd.set_detect_anomaly(True)

    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = os.path.join(data_root, "OPV2V/train")
    hypes["validate_dir"] = os.path.join(data_root, "OPV2V/test")
    dataset = build_dataset(hypes, visualize=True, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    attck_config = yaml_utils.load_yaml(
        "/workspace/AdvCollaborativePerception/uap/configs/uap.yaml", None
    )
    uap = UniversalAttacker(attck_config, device)
    uap.patch_obj.init()
    attacker = PatchAttack(
        norm="L_infty",
        device=device,
        cfg=attck_config["ATTACKER"],
        detector_attacker=uap,
    )

    V2VAM_model = uap.detectors["pointpillar_V2VAM"]
    v2vnet_model = uap.detectors["pointpillar_v2vnet"]
    model_dir = os.path.join(model_root, "pointpillar_V2VAM")
    for i, batch_data in tqdm(enumerate(data_loader)):
        if i >= 1:
            break
        with torch.no_grad():
            output_dict = OrderedDict()
            batch_data = train_utils.to_device(batch_data, device)
            attack_dict = {"attack_tagget": "remove"}
            gt_box_tensor = V2VAM_model.post_processor.generate_gt_bbx(batch_data)
            cav_content = batch_data["ego"]
            cav_content["gt_bboxes"] = gt_box_tensor
            cav_content["origin_lidar"] = cav_content["origin_lidar"][0].cpu().numpy()
            cav_content["lidar_pose"] = cav_content["lidar_pose"].cpu().numpy()
            data_dict = V2VAM_model.feature_encoder(cav_content)

        features = []
        j = 0
        for obj_id in tqdm(cav_content["object_ids"]):
            if j >= 10:
                break
            feature_index, obj_bbox = get_feature_index(cav_content, dataset, obj_id)
            attack_dict["obj_idx"] = feature_index
            attack_dict["object_bbox"] = obj_bbox
            loss, pred_box_tensor, pred_score = attacker.attack(
                data_dict, V2VAM_model, attack_dict
            )

            # vis_save_path = os.path.join(result_path, 'attack_%05d.png' % obj_id)
            # draw_attack(cav_content, pred_box_tensor, save=vis_save_path)
            j += 1

    patch_path = os.path.join(result_path, "patch")
    os.makedirs(patch_path, exist_ok=True)
    save_file = os.path.join(patch_path, "{}.pth".format("patch_{}_data".format({i})))
    uap.patch_obj.save(save_file)
