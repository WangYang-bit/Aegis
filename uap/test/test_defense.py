import argparse
import os
import sys
import time

import torch
from attaker import UniversalAttacker
from config import data_root, model_root, uap_root
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(uap_root)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from tools.feature_analys import *
from utils import eval_utils, train_utils
from utils.eval_utils import caluclate_tp_fp_expt_attack, eval_attack
from utils.visualizor import draw_attack

result_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "result")
os.makedirs(result_path, exist_ok=True)
attack_config_file = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "configs/uap.yaml"
)

train_path = os.path.join(data_root, "train")
test_path = os.path.join(data_root, "test")


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--hypes_yaml",
        type=str,
        required=True,
        help="data generation yaml file needed ",
    )
    parser.add_argument("--model_dir", default="", help="Continued training path")
    parser.add_argument("--init_model_dir", default="", help="init model path")
    parser.add_argument(
        "--half", action="store_true", help="whether train with half precision."
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    opt = parser.parse_args()
    return opt


def attack_visulize(attack_config, sence_id, obj_id=None):
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")

    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = train_path
    hypes["validate_dir"] = test_path
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

    uap = UniversalAttacker(attack_config, device)
    uap.init_attaker()
    apply_patch = attack_config["eval_params"]["apply_attack"]

    if not attack_config["debug"]:
        saved_path = eval_utils.setup_eval(
            attack_config, "visulize/" + attack_config["patch"]["name"]
        )

    for i, batch_data in tqdm(enumerate(data_loader)):
        if i == sence_id:
            break
    batch_data = train_utils.to_device(batch_data, device)
    gt_box_tensor = dataset.post_processor.generate_gt_bbx(batch_data)
    cav_content = batch_data["ego"]
    cav_content["gt_bboxes"] = gt_box_tensor
    cav_content["origin_lidar"] = cav_content["origin_lidar"][0].cpu().numpy()
    cav_content["lidar_pose"] = cav_content["lidar_pose"].cpu().numpy()

    if obj_id is None:
        obj_index_dict = get_random_index(cav_content, dataset, 1)
        obj_id = list(obj_index_dict.keys())[0]
        feature_index, obj_bbox = obj_index_dict[obj_id]
    else:
        feature_index, obj_bbox = get_feature_index(cav_content, dataset, obj_id)

    for detector_name, detector in uap.detectors.items():
        with torch.no_grad():
            cav_content["anchor_box"] = torch.from_numpy(
                detector.post_processor.generate_anchor_box()
            )

            data_dict = detector.feature_encoder(cav_content)

            spatial_feature = data_dict["spatial_features_2d"][0]

            attack_dict = {"attack_tagget": "remove"}
            attack_dict["obj_idx"] = feature_index
            attack_dict["object_bbox"] = obj_bbox

            clean_output = uap.attack(data_dict, detector, attack_dict, False)
            output_dict = uap.attack(data_dict, detector, attack_dict, apply_patch)

            if not attack_config["debug"]:
                vis_save_path = os.path.join(
                    saved_path,
                    "scene_%05d_%05d_%s.png" % (sence_id, obj_id, detector_name),
                )
                draw_attack(
                    cav_content,
                    output_dict["pred_box_tensor"],
                    clean_output["pred_box_tensor"],
                    save=vis_save_path,
                )
                if attack_config["visulize"]["save_feature"]:
                    feature_visualize(
                        spatial_feature,
                        save_file=os.path.join(
                            saved_path,
                            "feature_%05d_%s.png" % (sence_id, detector_name),
                        ),
                    )
                    feature_visualize(
                        output_dict["adv_feature"][0],
                        save_file=os.path.join(saved_path, "adv_feature.png"),
                    )
                    feature_visualize(
                        output_dict["fused_feature"][0],
                        save_file=os.path.join(saved_path, "fused_feature.png"),
                    )
    # draw_point_cloud(cav_content, save=os.path.join(saved_path, 'pointcloud.png'))
    if attack_config["visulize"]["save_patch"]:
        feature_visualize(
            uap.patch_obj.patch, save_file=os.path.join(saved_path, "patch.png")
        )


def eval_patch(attack_config):
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
    apply_patch = attack_config["eval_params"]["apply_attack"]

    if not attack_config["debug"]:
        saved_path = eval_utils.setup_eval(
            attack_config, "eval/" + attack_config["patch"]["name"]
        )

    for detector_name, detector in uap.detectors.items():
        attack_result = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
            "attack_success": [],
            "attack_failed_scene": [],
        }

        clean_result = {
            0.3: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.5: {"tp": [], "fp": [], "gt": 0, "score": []},
            0.7: {"tp": [], "fp": [], "gt": 0, "score": []},
        }

        detector.eval()
        print("evaluating patch on %s...." % detector_name)
        total_num = (
            attack_config["eval_params"]["batch_num"]
            * attack_config["eval_params"]["object_num"]
        )
        pbar2 = tqdm(total=total_num, leave=True)
        for i, batch_data in enumerate(data_loader):
            if i >= attack_config["eval_params"]["batch_num"]:
                break
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                attack_dict = {"attack_tagget": "remove"}
                gt_box_tensor = detector.post_processor.generate_gt_bbx(batch_data)
                cav_content = batch_data["ego"]
                cav_content["gt_bboxes"] = gt_box_tensor
                cav_content["anchor_box"] = torch.from_numpy(
                    detector.post_processor.generate_anchor_box()
                )
                data_dict = detector.feature_encoder(cav_content)
                obj_index_dict = get_random_index(
                    cav_content, dataset, attack_config["eval_params"]["object_num"]
                )
                j = 0

                clean_output = uap.attack(data_dict, detector, attack_dict, False)

                for obj_id, (feature_index, obj_bbox) in obj_index_dict.items():
                    start_time = time.time()
                    attack_dict["obj_idx"] = feature_index
                    attack_dict["object_bbox"] = obj_bbox
                    attack_output = uap.attack(
                        data_dict, detector, attack_dict, apply_patch
                    )

                    attack_success = eval_attack(
                        attack_output["pred_box_tensor"],
                        clean_output["pred_box_tensor"],
                        attack_dict,
                        attack_config["iou_thresh"],
                    )
                    attack_result["attack_success"].append(1 if attack_success else 0)
                    # if not attack_success:
                    #     attack_result['attack_failed_scene'].append("%d_%d" % (i, obj_id))

                    caluclate_tp_fp_expt_attack(
                        attack_output["pred_box_tensor"],
                        attack_output["pred_score"],
                        gt_box_tensor,
                        attack_result,
                        attack_dict,
                        0.3,
                    )
                    caluclate_tp_fp_expt_attack(
                        attack_output["pred_box_tensor"],
                        attack_output["pred_score"],
                        gt_box_tensor,
                        attack_result,
                        attack_dict,
                        0.5,
                    )
                    caluclate_tp_fp_expt_attack(
                        attack_output["pred_box_tensor"],
                        attack_output["pred_score"],
                        gt_box_tensor,
                        attack_result,
                        attack_dict,
                        0.7,
                    )

                    caluclate_tp_fp_expt_attack(
                        clean_output["pred_box_tensor"],
                        clean_output["pred_score"],
                        gt_box_tensor,
                        clean_result,
                        attack_dict,
                        0.3,
                    )
                    caluclate_tp_fp_expt_attack(
                        clean_output["pred_box_tensor"],
                        clean_output["pred_score"],
                        gt_box_tensor,
                        clean_result,
                        attack_dict,
                        0.5,
                    )
                    caluclate_tp_fp_expt_attack(
                        clean_output["pred_box_tensor"],
                        clean_output["pred_score"],
                        gt_box_tensor,
                        clean_result,
                        attack_dict,
                        0.7,
                    )

                    pbar2.set_description(
                        f"[model {detector_name}][{(i*attack_config['eval_params']['object_num'])+ j + 1}/{total_num}], || result: {'success ' if attack_result else 'failed '}"
                    )
                    pbar2.update(1)
                    j += 1

        if not attack_config["debug"]:
            eval_utils.eval_final_result(
                detector_name, attack_result, clean_result, saved_path
            )


if __name__ == "__main__":
    # opt = train_parser()
    attack_config = yaml_utils.load_yaml(attack_config_file, None)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    # print(sys.path)
    eval_patch(attack_config)
    # attack_visulize(attack_config, 20, 1075)
