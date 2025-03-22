import argparse
import os
import pickle
import sys
import time


import torch
from uap.attaker import UniversalAttacker
from uap.config import data_root, model_root, uap_root
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(uap_root)

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from uap.tools.feature_analys import *
from uap.utils import eval_utils, train_utils
from uap.utils.eval_utils import caluclate_tp_fp_expt_attack, eval_attack
from uap.utils.visualizor import draw_attack
from opencood.utils import box_utils, common_utils
from uap.utils.data_utils import compute_index_in_scenario
from uap.config import data_root, uap_root, uap_path, model_root, len_record
from uap.attaker import UniversalAttacker
from uap.utils.box_utils import compute_iou
from uap.defense.defender import spatial_temporal_defender, get_car_position
from uap.defense.robosac_defender import robosac_defender
from uap.defense.robust_defender import robust_defender
from uap.defense.mvp_defender import mvp_defender

result_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "result")
os.makedirs(result_path, exist_ok=True)
attack_config_file = os.path.join(
    os.path.abspath(os.path.dirname(__file__)), "configs/uap.yaml"
)

train_path = os.path.join(data_root, "train")
test_path = os.path.join(data_root, "test")

def convert_numpy(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_numpy(x) for x in obj)
    else:
        return obj
    
def get_spoof_boxes(spoof_boxes, num, scene_id, timestamp):
    if scene_id not in spoof_boxes:
        return None
    if timestamp not in spoof_boxes[scene_id]:
        return None
    spoof_boxes = spoof_boxes[scene_id][timestamp]
    if len(spoof_boxes) > num:
        spoof_boxes = spoof_boxes[:num]
    return spoof_boxes

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

    # for i, batch_data in tqdm(enumerate(data_loader)):
    #     if i == sence_id :
    #         break
    data_dict = dataset.__getitem__(sence_id)
    batch_data = dataset.collate_batch_test([data_dict])
    batch_data = train_utils.to_device(batch_data, device)

    gt_box_tensor, object_ids = dataset.post_processor.generate_gt_bbx(batch_data)
    cav_content = batch_data["ego"]

    cav_content["lidar_pose"] = np.array(cav_content["lidar_pose"])

    cav_content["gt_bboxes"] = gt_box_tensor
    cav_content["origin_lidar"] = cav_content["origin_lidar"].cpu().numpy()
    attack_mode = attack_config['eval_params']['attack_mode']
    if obj_id is None:
        obj_index_dict = get_random_index(cav_content, dataset, 1, mode=attack_mode)
        obj_id = list(obj_index_dict.keys())[0]
        feature_index, obj_bbox = obj_index_dict[obj_id]
    else:
        feature_index, obj_bbox = get_feature_index(cav_content, dataset, obj_id)



    for detector_name, detector in uap.detectors.items():
        with torch.no_grad():
            cav_content["anchor_box"] = torch.from_numpy(
                detector.post_processor.generate_anchor_box()
            )
            cav_content["attack_bbox"] = obj_bbox
            data_dict = detector.feature_encoder(cav_content)

            spatial_feature = data_dict["spatial_features_2d"][0]

            attack_dict = {"attack_tagget": attack_mode }
            attack_dict["obj_idx"] = feature_index
            attack_dict["object_bbox"] = obj_bbox

            clean_output = uap.attack(data_dict, detector, attack_dict, False)
            output_dict = uap.attack(data_dict, detector, attack_dict, apply_patch)

            if not attack_config["debug"]:
                vis_save_path = os.path.join(
                    saved_path,
                    "scene_%05d_%05d_%s_%s.png" % (sence_id, obj_id, detector_name,attack_mode),
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


def train_patch(attack_config):
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")

    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = train_path
    hypes["validate_dir"] = test_path
    dataset = build_dataset(hypes, visualize=False, train=True)

    train_loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=16,
        collate_fn=dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )

    uap = UniversalAttacker(attack_config, device)

    num_steps = len(train_loader)
    if not attack_config["debug"]:
        saved_path = train_utils.setup_train(
            attack_config, "train/%s_" % (attack_config["patch"]["name"])
        )
        # record training
        writer = SummaryWriter(saved_path)

    attack_dict = {"attack_tagget": attack_config["attack"]["mode"]}
    for epoch in range(attack_config["train_params"]["max_epoch"]):
        pbar2 = tqdm(total=attack_config["train_params"]["batch_num"], leave=True)

        for i, batch_data in enumerate(train_loader):
            if i >= attack_config["train_params"]["batch_num"]:
                break
            batch_data = train_utils.to_device(batch_data, device)
            cav_content = batch_data["ego"]
            batch_loss = []
            obj_index_dict = get_random_index(
                cav_content,
                dataset,
                attack_config["eval_params"]["object_num"],
                attack_config["attack"]["mode"],
            )
            for obj_id, (feature_index, obj_bbox) in obj_index_dict.items():
                attack_dict["obj_idx"] = feature_index
                attack_dict["object_bbox"] = obj_bbox.to(device)
                loss = uap.attack_train(cav_content, attack_dict)
                batch_loss.append(loss)
            if not attack_config["debug"]:
                train_utils.logging(
                    torch.tensor(batch_loss).mean(),
                    epoch,
                    i,
                    attack_config["train_params"]["batch_num"],
                    writer,
                    pbar=pbar2,
                )
            pbar2.update(1)

        if (
            epoch % attack_config["train_params"]["save_freq"] == 0
            and not attack_config["debug"]
        ):
            patch_save_file = os.path.join(
                saved_path,
                "%s_patch_%d_model_epoch%d.pth"
                % (attack_config["attack"]["mode"], uap.get_detectors_num(), epoch + 1),
            )
            uap.patch_obj.save(patch_save_file)


def eval_attacks(attack_config):
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")
    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = train_path
    hypes["validate_dir"] = test_path
    

    spoof_boxes_path = os.path.join(uap_path, "data/OPV2V/attack/spoof.pkl")
    with open(spoof_boxes_path, 'rb') as f:
        spoof_boxes = pickle.load(f)
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
    attack_mode = attack_config["eval_params"]["attack_mode"]
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
            # if i >= attack_config["eval_params"]["batch_num"]:
            #     break
            # with torch.no_grad():
            batch_data = train_utils.to_device(batch_data, device)
            attack_dict = {"attack_tagget": attack_mode}
            gt_box_tensor, object_ids = dataset.post_processor.generate_gt_bbx(batch_data)
            cav_content = batch_data["ego"]
            cav_content["gt_bboxes"] = gt_box_tensor
            cav_content["anchor_box"] = torch.from_numpy(
                detector.post_processor.generate_anchor_box()
            )
            data_dict = detector.feature_encoder(cav_content)
            
            obj_index_dict = get_random_index(
                    cav_content, dataset, attack_config["eval_params"]["object_num"], mode= attack_mode
              )
            # if attack_mode == "spoof":
            #     obj_index_dict = {}
            #     spoof_boxes_list = get_spoof_boxes(spoof_boxes, attack_config["eval_params"]["object_num"], cav_content['scene_id'], cav_content['timestamp_id'])
            #     if spoof_boxes_list is None:
            #         continue
            #     for j, obj_bbox in enumerate(spoof_boxes_list):
            #         feature_index = get_feature_index(obj_bbox, dataset)
            #         obj_bbox = torch.from_numpy(obj_bbox)
            #         obj_index_dict[j] = (feature_index, obj_bbox)
            # elif attack_mode == "remove":
            #     obj_index_dict = get_random_index(
            #         cav_content, dataset, attack_config["eval_params"]["object_num"], mode= attack_mode
            #     )
            j = 0
            with torch.no_grad():
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
                    attack_config["iou_threshold"],
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
                    f"[model {detector_name}][{(i * attack_config['eval_params']['object_num']) + j + 1}/{total_num}], || result: {'success ' if attack_success else 'failed '}"
                )
                pbar2.update(1)
                j += 1

        if not attack_config["debug"]:
            eval_utils.eval_final_result(
                detector_name, attack_result, clean_result, saved_path
            )
        print("attack success rate:", sum(attack_result['attack_success'])/len(attack_result['attack_success']))

def eval_defense(config):
    if config['defense_params']['name'] == "spatial_temporal":
        defender = spatial_temporal_defender(config=config, device=device)
        print("Init defender spatial_temporal_defender...")
    elif config['defense_params']['name'] == "robosac":
        defender = robosac_defender(config=config, device=device)
        FP = 0
        print("Init defender robosac_defender...")
    elif config['defense_params']['name'] == "mvp":
        defender = mvp_defender(device=device)
        print("Init defender mvp_defender...")
    elif config['defense_params']['name'] == "robust":
        defender = robust_defender(config=config, device=device)
        FP = 0
        print("Init defender robust_defender...")
    else:
        print("Unrecongnize defense name!")
        return 
    
    print("---------------Start evaluating defense...-----------------")
    print("---------------Loading data...-----------------")
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")
    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = os.path.join(data_root, "train")
    hypes["validate_dir"] = os.path.join(data_root, "test")
    v2v_dataset = build_dataset(hypes, visualize=False, train=False)
    data_loader = DataLoader(
        v2v_dataset,
        batch_size=1,
        num_workers=8,
        collate_fn=v2v_dataset.collate_batch_test,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
    )
    
    print("---------------Loading detector and attacker...---------------")
    uap = UniversalAttacker(config, device)
    fusion_detector = uap.detectors["pointpillar_V2VAM"]
    if config['defense_params']['name'] == "spatial_temporal":
        single_cav_detector = uap.detectors["pointpillar_single_car_large"]
    apply_attack = config["eval_params"]["apply_attack"]
    attack_mode = config['eval_params']['attack_mode']
    defense_results = {
            "spoof_error": [],
            "spoof_label": [],
            "spoof_location": [],
            "remove_error": [],
            "remove_label": [],
            "remove_location": [],
            "success": [],
    }
    final_result = { "total_attacker_num": 0 }
    total_frame = 0
    attack_frame = 0
    attack_success = []
    world_model_time = []
    consist_time = []
    total_cav_num = 0
    for sence_id, batch_data in tqdm(enumerate(data_loader)):
        # if sence_id > 30:
        #     break
        index, scenario_id = compute_index_in_scenario(sence_id, len_record)
        if config['defense_params']['name'] == "spatial_temporal" and index < defender.condition_frame:
            continue
        total_frame += 1
        v2v_batch_data = train_utils.to_device(batch_data, device)
        gt_box_tensor, gt_obj_id = fusion_detector.post_processor.generate_gt_bbx(v2v_batch_data)
        cav_content = v2v_batch_data["ego"]
        
        data_dict = fusion_detector.feature_encoder(cav_content)
        fusion_anchor_box = torch.from_numpy(
            fusion_detector.post_processor.generate_anchor_box()
        ).to(device)
        data_dict["anchor_box"] = fusion_anchor_box
        
        obj_index_dict = get_random_index(cav_content, v2v_dataset, 1, mode=attack_mode)
        obj_id = list(obj_index_dict.keys())[0]
        feature_index, obj_bbox = obj_index_dict[obj_id]
        attack_dict = {"attack_tagget": attack_mode}
        attack_dict["obj_idx"] = feature_index
        attack_dict["object_bbox"] = obj_bbox
        feature = data_dict["spatial_features_2d"]
        output_dict = uap.attack(data_dict, fusion_detector, attack_dict=attack_dict, apply_attack=True)
        data_dict['spatial_features_2d'] = output_dict['adv_feature']
        obj_bbox_corners = box_utils.boxes_to_corners_3d(obj_bbox.unsqueeze(0), order=fusion_detector.post_processor.params['order'])[0]
        single_cav_output = None
        cav_position = None
        if config['defense_params']['name'] == "spatial_temporal":
            single_cav_output = uap.attack_single_car(data_dict, single_cav_detector, attack_dict=None, apply_attack=False)
            cav_position = cav_content['cav_pose']

        iou_attack = compute_iou(obj_bbox_corners, output_dict["pred_box_tensor"])
        if attack_mode == 'remove' and len(iou_attack)>0 and iou_attack.max() > 0.1:
            attack_success.append(0)
            continue
        elif attack_mode == 'spoof' and len(iou_attack)>0 and iou_attack.max() < 0.3:
            attack_success.append(0)
            continue
        attack_frame += 1
        attack_success.append(1)
        sence_case = {
            'sence_id': sence_id,
            'scenario_index': scenario_id,
            'data_dict': data_dict,
            'fusion_feature': output_dict['fused_feature'],
            "fusion_box": output_dict['pred_box_tensor'],
            'fusion_anchor_box': fusion_anchor_box,
            'fusion_output': output_dict,
            "gt_bboxes": gt_box_tensor,
            "ego_lidar_pose": cav_content["lidar_pose"],
            'single_cav_anchor_box': fusion_anchor_box,
            'single_cav_output': single_cav_output,
            'car_position': cav_position,
            "attack_mode" : attack_mode,
            "attack_bbox": obj_bbox,
            "attack_bbox_corners": obj_bbox_corners,
            "init_feature": feature,
            }
        if config['defense_params']['name'] == "spatial_temporal":
            result = defender.defense(sence_case, fusion_detector)
            world_model_time.append(result['world_model_time'])
            consist_time.append(result['consistent_time'])
            defense_results = defender.defense_evaluation_processor(result, sence_case, defense_results)
        if config['defense_params']['name'] == "robosac":
            result = defender.defense(sence_case, detector=fusion_detector)
            defense_results = defender.defense_evaluation_processor(result, sence_case, defense_results)
            # attacker_list = result["attacker_list"]
            # defense_results["total_attacker_num"] += len(attacker_list)
            # total_cav_num += len(cav_content["cav_list"]) - 1
            # defense_results["success"].append(0 in attacker_list)
            # if 0 in attacker_list:
            #     if len(attacker_list) > 1:
            #         FP += len(attacker_list) - 1
            # elif len(attacker_list) > 0:
            #     FP += len(attacker_list)
        if config['defense_params']['name'] == "mvp":
            start_time = time.time()
            vehicle_metrics =  defender.defense(sence_case)
            end_time = time.time()
            # print("defense time",end_time - start_time)
            if vehicle_metrics is not None:
                defense_results = defender.defense_evaluation_processor(vehicle_metrics, sence_case, defense_results,iou_thres=config['iou_threshold'])
        if config['defense_params']['name'] == "robust":
            result = defender.defense(sence_case, fusion_detector)
            defense_results = defender.defense_evaluation_processor(result, sence_case, defense_results)

    for key, data in defense_results.items():
        if isinstance(data, list) and len(data) > 0:
            defense_results[key] = np.concatenate(data).reshape(-1)
    if attack_mode == "remove":
        best_TPR, best_FPR, roc_auc, best_thres , tpr_data, fpr_data = defender.compute_roc(defense_results["remove_error"], defense_results["remove_label"])
    elif attack_mode == "spoof":
        best_TPR, best_FPR, roc_auc, best_thres , tpr_data, fpr_data = defender.compute_roc(defense_results["spoof_error"], defense_results["spoof_label"])
    final_result.update({"total_frame": total_frame})
    final_result.update({"attack_success_rate": attack_frame/total_frame})
    final_result.update({"attack_frame": attack_frame})
    final_result.update({"success_rate": np.mean(defense_results["success"])})
    final_result.update({"FP_rate": best_FPR})
    final_result.update({"best_TPR": best_TPR})
    final_result.update({"roc_auc": roc_auc})
    final_result.update({"best_thres": best_thres})
    final_result.update({"tpr_data": tpr_data})
    final_result.update({"fpr_data": fpr_data})
    final_result.update({"world_model_time": np.mean(world_model_time)})
    final_result.update({"consistent_time": np.mean(consist_time)})
    print("attack success rate:", attack_frame/total_frame)
    print("world_model_time:", np.mean(world_model_time))
    print("consistent_time:", np.mean(consist_time))
    print("success rate: ", np.mean(defense_results["success"]))
    print("FP rate:",best_FPR)
    print("TP rate:", best_TPR)

    # elif config['defense_params']['name'] == "robosac":
    #     final_result.update({"total_frame": total_frame})
    #     final_result.update({"attack_success_rate": attack_frame/total_frame})
    #     final_result.update({"attack_frame": attack_frame})
    #     final_result.update({"success_rate": np.mean(defense_results["success"])})
    #     final_result.update({"FP_rate": FP/(total_cav_num - attack_frame + 1)})
    #     final_result.update({"TP_rate": np.mean(defense_results["success"])})
    #     print("success rate: ", np.mean(defense_results["success"]))
    #     print("FP rate:",FP/(total_cav_num - attack_frame + 1))
    #     print("TP rate:", np.mean(defense_results["success"]))
    # else:
    #     final_result.update({"total_frame": total_frame})
    #     final_result.update({"attack_success_rate": attack_frame/total_frame})
    #     final_result.update({"attack_frame": attack_frame})
    #     final_result.update({"success_rate": defense_results["success"]/attack_frame})
    #     final_result.update({"FP_rate": defense_results["FP"]/attack_frame})
    #     print("success rate: ", defense_results["success"]/attack_frame)
    #     print("FP rate:",defense_results["FP"]/attack_frame)

    save_path = os.path.join("/home/UAP_attack/uap/logs/defense",config['defense_params']['name'] + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()))
    os.makedirs(save_path, exist_ok=True)
    output_file = "defense_result.yaml"
    final_result_converted = convert_numpy(final_result)
    yaml_utils.save_yaml(final_result_converted, os.path.join(save_path, output_file))
    yaml_utils.save_yaml(config, os.path.join(save_path, "config.yaml"))
    print("defense result saved to ", os.path.join(save_path, output_file))
    print("evaluation done!")

if __name__ == "__main__":
    # opt = train_parser()
    attack_config = yaml_utils.load_yaml(attack_config_file, None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(sys.path)

    # train_patch(attack_config)
    # eval_attacks(attack_config)
    # attack_visulize(attack_config, 1)
    eval_defense(attack_config)
    # Print peak memory usage
    # print(
    #     f"Peak memory usage: {torch.cuda.max_memory_allocated(device) / 1024**2:.2f} MB"
    # )
