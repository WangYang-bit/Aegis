import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm



import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from uap.attaker import UniversalAttacker
from uap.config import data_root, model_root, uap_root
from uap.tools.feature_analys import *
from uap.utils import train_utils

train_path = os.path.join(data_root, "train")
test_path = os.path.join(data_root, "test")


def collect_fusion_feature(attack_config, device, save_path=None):
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")

    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = train_path
    hypes["validate_dir"] = test_path
    dataset = build_dataset(hypes, visualize=False, train=False)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    uap = UniversalAttacker(attack_config, device)
    uap.init_attaker()
    apply_patch = attack_config["eval_params"]["apply_attack"]

    for i, batch_data in tqdm(enumerate(data_loader)):
        batch_data = train_utils.to_device(batch_data, device)

        gt_box_tensor, object_ids = dataset.post_processor.generate_gt_bbx(batch_data)
        cav_content = batch_data["ego"]

        cav_content["lidar_pose"] = np.array(cav_content["lidar_pose"])

        cav_content["gt_bboxes"] = gt_box_tensor

        for detector_name, detector in uap.detectors.items():
            with torch.no_grad():
                cav_content["anchor_box"] = torch.from_numpy(
                    detector.post_processor.generate_anchor_box()
                )

                data_dict = detector.feature_encoder(cav_content)
                attack_dict = {"attack_tagget": "remove"}

                clean_output = uap.attack(data_dict, detector, attack_dict, False)

                if not attack_config["debug"]:
                    scene_id = cav_content["scene_id"]
                    timestamp_index = cav_content["timestamp_index"]
                    file_path = os.path.join(save_path, f"{scene_id}")
                    os.makedirs(file_path, exist_ok=True)
                    file_path = os.path.join(file_path, f"{timestamp_index}.pth")
                    # compressed_data = {
                    #     "scene_id": scene_id,
                    #     "fusion_feature": clean_output["fused_feature"][0].cpu(),
                    #     "label_dict": {
                    #         k: v[0].cpu() for k, v in data_dict["label_dict"].items()
                    #     },
                    # }
                    # with gzip.open(file_path, "wb") as f:
                    #     torch.save(compressed_data, f)

                    # with gzip.open(file_path, "rb") as f:
                    #     loaded_data = torch.load(f)

                    # scene_id = loaded_data["scene_id"]
                    # fusion_feature = loaded_data["fusion_feature"]
                    # label_dict = loaded_data["label_dict"]

                    torch.save(
                        {
                            "scene_id": scene_id,
                            "fusion_feature": clean_output["fused_feature"][0].cpu(),
                            "label_dict": {
                                k: v[0].cpu()
                                for k, v in data_dict["label_dict"].items()
                            },
                        },
                        file_path,
                    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    attack_config_file = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "../configs/uap.yaml"
    )
    attack_config = yaml_utils.load_yaml(attack_config_file, None)
    result_path = os.path.join(
        data_root, "fusion_feature/test"
    )
    os.makedirs(result_path, exist_ok=True)

    collect_fusion_feature(attack_config, device, result_path)
