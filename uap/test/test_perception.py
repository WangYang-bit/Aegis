import os
from collections import OrderedDict

import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
import torch
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
from opencood.utils import eval_utils
from tools.feature_analys import *
from torch.utils.data import DataLoader
from tqdm import tqdm

from uap.attaker import UniversalAttacker
from uap.config import data_root, model_root, uap_root

train_path = os.path.join(data_root, "train")
test_path = os.path.join(data_root, "test")


def test_fusion(detector_name):
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")
    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = os.path.join(data_root, "OPV2V/train")
    hypes["validate_dir"] = os.path.join(data_root, "OPV2V/test")
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

    attck_config = yaml_utils.load_yaml(
        "/workspace/AdvCollaborativePerception/uap/configs/uap.yaml", None
    )
    uap = UniversalAttacker(attck_config, device)
    uap.patch_obj.init()

    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    encoder = uap.detectors["pointpillar_V2VAM"]
    fusion_model = uap.detectors[detector_name]
    for i, batch_data in tqdm(enumerate(data_loader)):
        # if i > 10:
        #     break
        with torch.no_grad():
            output_dict = OrderedDict()
            batch_data = train_utils.to_device(batch_data, device)
            gt_box_tensor = fusion_model.post_processor.generate_gt_bbx(batch_data)
            cav_content = batch_data["ego"]
            data_dict = encoder.feature_encoder(cav_content)
            output_dict["ego"] = fusion_model.fusion_decoder(data_dict)
            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
                batch_data, output_dict
            )

            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5
            )

            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7
            )
    save_path = os.path.join(uap_root, "uap/result/test_perception")

    os.makedirs(save_path, exist_ok=True)
    eval_utils.eval_final_results(result_stat, save_path, False)

    print("AP0.5: ", np.mean(result_stat[0.5]["score"]))
    print("AP0.7: ", np.mean(result_stat[0.7]["score"]))


def test_perception(detector_name):
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
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

    attck_config = yaml_utils.load_yaml(
        "/workspace/AdvCollaborativePerception/uap/configs/uap.yaml", None
    )
    uap = UniversalAttacker(attck_config, device)
    uap.init_attaker()

    result_stat = {
        0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
        0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
    }

    model = uap.detectors[detector_name]
    for i, batch_data in tqdm(enumerate(data_loader)):
        # if i > 10:
        #     break
        with torch.no_grad():
            output_dict = OrderedDict()
            batch_data = train_utils.to_device(batch_data, device)
            gt_box_tensor = model.post_processor.generate_gt_bbx(batch_data)
            cav_content = batch_data["ego"]
            cav_content["anchor_box"] = torch.from_numpy(
                model.post_processor.generate_anchor_box()
            )
            data_dict = model.feature_encoder(cav_content)
            output_dict["ego"] = model.fusion_decoder(data_dict)
            pred_box_tensor, pred_score, gt_box_tensor = dataset.post_process(
                batch_data, output_dict
            )

            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.5
            )

            eval_utils.caluclate_tp_fp(
                pred_box_tensor, pred_score, gt_box_tensor, result_stat, 0.7
            )
    save_path = os.path.join(uap_root, "uap/result/test_perception")

    os.makedirs(save_path, exist_ok=True)
    eval_utils.eval_final_results(result_stat, save_path, False)

    print("AP0.5: ", np.mean(result_stat[0.5]["score"]))
    print("AP0.7: ", np.mean(result_stat[0.7]["score"]))


if __name__ == "__main__":
    detector_name = "pointpillar_v2vnet"
    test_perception(detector_name)
    # test_fusion(detector_name)
