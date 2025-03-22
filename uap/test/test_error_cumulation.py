import time

import numpy as np
import opencood.hypes_yaml.yaml_utils as yaml_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets.time_series_dataset import TimeSeriesDataset
from einops import rearrange
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.utils import box_utils
from tqdm import tqdm
from utils import train_utils

import wandb
from uap.models.world_model import WorldModel, AR_Loss




if __name__ == "__main__":
    config_file = "/home/UAP_attack/uap/configs/uap.yaml"
    config = yaml_utils.load_yaml(config_file, None)
    args = config["model"]["world_model"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_params = config["data"]

    print("-----------------Dataset Building------------------")
    val_dataset = TimeSeriesDataset(dataset_params, train=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=16,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    print("---------------Creating Model------------------")
    model = WorldModel(args, device=device)
    if config["train_params"]["init_model_path"] is not None:
        model = WorldModel(args, device=device)
        init_epoch, model = train_utils.load_saved_model(
            config["train_params"]["init_model_path"], model
        )
    else:
        init_epoch = 0
    criterion = AR_Loss(
        alpha=config["AR_loss"]["args"]["mse_weight"],
        beta=config["AR_loss"]["args"]["task_weight"],
    )
    
    model.to(device)
    sence_id = 0
    predict_steps = 10
    target_list = []
    with torch.no_grad():
        sence_id = sence_id + 1
        data_dict = val_dataset.__getitem__(sence_id)
        batch_data = val_dataset.collate_fn([data_dict])
        batch_data = train_utils.to_device(batch_data, device)
        
        for i in range(predict_steps):
            sence_id = sence_id + 1
            data_dict = val_dataset.__getitem__(sence_id)
            batch_data = val_dataset.collate_fn([data_dict])
            batch_data = train_utils.to_device(batch_data, device)
            target = {
                  "target_feature": batch_data["target_feature"],
                  "label_dict": batch_data["label_dict"],
              }
            target_list.append(target)
            if i == 0:
                output = model(batch_data)
                loss = criterion(output, target)
            else:
                batch_data["history_feature"] = output["history_feature"]
            pre_batch_data = batch_data


