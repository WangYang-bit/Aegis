import os
import sys

import torch

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from uap.datasets.time_series_dataset import TimeSeriesDataset
from uap.config import data_root, uap_root, uap_path, model_root
from uap.tools.feature_analys import *
from uap.utils import eval_utils, train_utils
from uap.utils.visualizor import draw_bbox,draw_attack
from uap.models.world_model import WorldModel
from uap.attaker import UniversalAttacker

def test_ar(config, sence_id):
    print("-----------------Dataset Building------------------")
    dataset_params = config["data"]
    dataset = TimeSeriesDataset(dataset_params, train=False)
    
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")

    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = os.path.join(data_root, "train")
    hypes["validate_dir"] = os.path.join(data_root, "test")
    v2v_dataset = build_dataset(hypes, visualize=True, train=False)


    if not config["debug"]:
        saved_path = eval_utils.setup_eval(
            config, "visulize/ar_scene_%05d" % (sence_id)
        )

    model_args = config["model"]["world_model"]
    print("---------------Creating Model------------------")
    model = WorldModel(model_args, device=device)
    assert config["train_params"]["init_model_path"] is not None, "model path is None"
    init_epoch, model = train_utils.load_saved_model(
        config["train_params"]["init_model_path"], model
    )

    model.to(device)
    
    uap = UniversalAttacker(config, device)
    detector = uap.detectors["pointpillar_V2VAM"]
    # for i, batch_data in tqdm(enumerate(data_loader)):
    #     if i == sence_id :
    #         break
    data_dict = dataset.__getitem__(sence_id)
    batch_data = dataset.collate_fn([data_dict])
    batch_data = train_utils.to_device(batch_data, device)

    v2v_data_dict = v2v_dataset.__getitem__(sence_id + (batch_data["scenario_index"][0]+1)*dataset.condition_frames)
    v2v_batch_data = v2v_dataset.collate_batch_test([v2v_data_dict])
    v2v_batch_data = train_utils.to_device(v2v_batch_data, device)

    gt_box_tensor, object_ids = v2v_dataset.post_processor.generate_gt_bbx(v2v_batch_data)
    cav_content = v2v_batch_data["ego"]

    cav_content["lidar_pose"] = np.array(cav_content["lidar_pose"])

    cav_content["gt_bboxes"] = gt_box_tensor
    cav_content["origin_lidar"] = cav_content["origin_lidar"].cpu().numpy()

    anchor_box = detector.post_processor.generate_anchor_box()
    batch_data["anchor_box"] = torch.from_numpy(np.array(anchor_box)).to(device)

    with torch.no_grad():

        target_feature = batch_data["target_feature"]
        psm = detector.cls_head(target_feature)
        rm = detector.reg_head(target_feature)
        gt_output = {'psm': psm, 'rm': rm}
        detect_box, pre_score = detector.post_processor.post_process_single_car(batch_data, gt_output)
        # gt_box = gt_box[0].cpu().numpy()

        output_dict = model.forward(batch_data)
        pre_feature = output_dict["pred_features"]
        pre_boxes, pre_score = detector.post_processor.post_process_single_car(batch_data, output_dict)
        # pre_boxes = pre_boxes[0].cpu().numpy()

        if not config["debug"]:
            vis_save_path = os.path.join(
                saved_path,
                "ar_scene_%05d.png" % (sence_id),
            )
            draw_attack(
                cav_content,
                pre_boxes[0],
                gt_box_tensor,
                save=os.path.join(
                saved_path,
                "ar_scene_%05d_pred.png" % (sence_id),
            )
            )
            draw_attack(
                cav_content,
                detect_box[0],
                gt_box_tensor,
                save=os.path.join(
                saved_path,
                "ar_scene_%05d_detect.png" % (sence_id),
            )
            )
            if config["visulize"]["save_feature"]:
                feature_visualize(
                    target_feature,
                    save_file=os.path.join(
                        saved_path,
                        "gt_feature.png",
                    ),
                )
                feature_visualize(
                    pre_feature,
                    save_file=os.path.join(saved_path, "pre_feature.png"),
                )


if __name__ == "__main__":
    # opt = train_parser()
    result_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "result")
    os.makedirs(result_path, exist_ok=True)
    config_file = os.path.join(
        uap_path, "configs/uap.yaml"
    )
    train_path = os.path.join(data_root, "train")
    test_path = os.path.join(data_root, "test")

    config = yaml_utils.load_yaml(config_file, None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_ar(config, 10)
