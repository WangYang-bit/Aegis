import torch
import torch.nn.functional as F
import numpy as np
import os, sys
import math

from collections import OrderedDict
from torch.utils.data import DataLoader
from tqdm import tqdm

from uap.attaker import UniversalAttacker
from patch import PatchManager
from attack.patch_attack import PatchAttack
from config import model_root,data_root, model_list,uap_root,third_party_root
opencood_root = os.path.join(third_party_root, "OpenCOOD")
sys.path.append(opencood_root)
sys.path.append(uap_root)


import third_party.OpenCOOD.opencood.hypes_yaml.yaml_utils as yaml_utils
from third_party.OpenCOOD.opencood.tools import train_utils, inference_utils
from third_party.OpenCOOD.opencood.data_utils.datasets import build_dataset
from third_party.OpenCOOD.opencood.utils import box_utils, eval_utils
from third_party.OpenCOOD.opencood.utils.pcd_utils import mask_points_by_range
from third_party.OpenCOOD.opencood.utils.transformation_utils import x1_to_x2
from third_party.OpenCOOD.opencood.utils.common_utils import torch_tensor_to_numpy

from uap.tools.attack_utils import inference_intermediate_fusion,attck_intermediate
from tools.feature_analys import*


if __name__ == "__main__":
    detectors = {}
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # model_dir = os.path.join(model_root, "pointpillar_attentive_fusion/config.yaml")
    # model_dir = os.path.join(model_root, "pointpillar_CoBEVT/config.yaml")
    model_dir = os.path.join(model_root, "pointpillar_V2VAM/config.yaml")
    # model_dir = os.path.join(model_root, "pointpillar_v2vnet/config.yaml")
    hypes = yaml_utils.load_yaml(model_dir, None)
    hypes["root_dir"] = os.path.join(data_root, "OPV2V/train")
    hypes["validate_dir"] = os.path.join(data_root, "OPV2V/test")
    dataset = build_dataset(hypes, visualize=False, train=False)
    print(f"{len(dataset)} samples found.")
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)
    
    attck_config = yaml_utils.load_yaml("/workspace/AdvCollaborativePerception/uap/configs/uap.yaml", None)
    uap = UniversalAttacker(attck_config,device)
    uap.patch_obj.init()
    attacker = PatchAttack( norm='L_infty', device=device, cfg=attck_config["ATTACKER"], detector_attacker=uap)
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
        detectors[detector_name] = detector
    
    result_stat = {0.3: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.5: {'tp': [], 'fp': [], 'gt': 0, 'score': []},                
                   0.7: {'tp': [], 'fp': [], 'gt': 0, 'score': []}}
    V2VAM_model = detectors["pointpillar_V2VAM"]
    v2vnet_model = detectors["pointpillar_v2vnet"]
    model_dir = os.path.join(model_root, "pointpillar_V2VAM")
    for i, batch_data in tqdm(enumerate(data_loader)):
        if i >= 1:
            break
        with torch.no_grad():
            output_dict = OrderedDict()
            batch_data = train_utils.to_device(batch_data, device)
            attack_dict = {'attack_tagget':'remove'}
            gt_box_tensor = V2VAM_model.post_processor.generate_gt_bbx(batch_data)
            cav_content = batch_data['ego']
            data_dict = V2VAM_model.feature_encoder(cav_content)
            features = []
            j = 0
            for obj_id in cav_content['object_ids']:
                if j >= 10:
                    break
                feature_index = get_feature_index(cav_content, dataset, obj_id)
                loss, pred_box_tensor, pred_score = attacker.attack(data_dict, V2VAM_model, attack_dict)
                feature = get_obj_feature(data_dict['spatial_features_2d'], feature_index, [10,10])
                features.append(feature)
                j += 1
            similarity_matrix = similarity_analys(features)
            print(similarity_matrix)