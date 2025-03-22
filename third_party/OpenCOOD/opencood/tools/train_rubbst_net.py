# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import os

import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils, train_utils
from uap.tools.feature_analys import get_random_index
from uap.attaker import UniversalAttacker
from uap.config import model_root, data_root

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


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    hypes["root_dir"] = train_path
    hypes["validate_dir"] = test_path
    multi_gpu_utils.init_distributed_mode(opt)

    print("-----------------Dataset Building------------------")
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=False, train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes["train_params"]["batch_size"], drop_last=True
        )

        train_loader = DataLoader(
            opencood_train_dataset,
            batch_sampler=batch_sampler_train,
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
        )
        # val_loader = DataLoader(opencood_validate_dataset,
        #                         sampler=sampler_val,
        #                         num_workers=8,
        #                         collate_fn=opencood_train_dataset.collate_batch_train,
        #                         drop_last=False)
    else:
        train_loader = DataLoader(
            opencood_train_dataset,
            batch_size=hypes["train_params"]["batch_size"],
            num_workers=8,
            collate_fn=opencood_train_dataset.collate_batch_train,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
        )
        # val_loader = DataLoader(opencood_validate_dataset,
        #                         batch_size=hypes['train_params']['batch_size'],
        #                         num_workers=8,
        #                         collate_fn=opencood_train_dataset.collate_batch_train,
        #                         shuffle=False,
        #                         pin_memory=False,
        #                         drop_last=True)

    print("---------------Creating Model------------------")
    model = train_utils.create_model(hypes)
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        init_epoch, model = train_utils.load_saved_model(saved_path, model)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)

    if opt.init_model_dir:
        init_hype_yaml = os.path.join(opt.init_model_dir, "config.yaml")
        init_hypes = yaml_utils.load_yaml(init_hype_yaml, None)
        init_model = train_utils.create_model(init_hypes)
        epoch, init_model = train_utils.load_saved_model(opt.init_model_dir, init_model)
        pillar_vfe_params = init_model.pillar_vfe.state_dict()
        model.pillar_vfe.load_state_dict(pillar_vfe_params)
        model.pillar_vfe.requires_grad_(False)
        scatter_params = init_model.scatter.state_dict()
        model.scatter.load_state_dict(scatter_params)
        model.scatter.requires_grad_(False)
        backbone_params = init_model.backbone.state_dict()
        model.backbone.load_state_dict(backbone_params)
        model.backbone.requires_grad_(False)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
    model_without_ddp = model

    if opt.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[opt.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # load the attacker
    attack_config = yaml_utils.load_yaml("/home/UAP_attack/uap/configs/uap.yaml", None)
    attacker = UniversalAttacker(attack_config, device)
    attacker.freeze_patch()
    attack_mode = attack_config["eval_params"]["attack_mode"]
    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler("cuda")

    print("Training start")
    epoches = hypes["train_params"]["epoches"]
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes["lr_scheduler"]["core_method"] != "cosineannealwarm":
            scheduler.step(epoch)
        if hypes["lr_scheduler"]["core_method"] == "cosineannealwarm":
            scheduler.step_update(epoch * num_steps + 0)
        for param_group in optimizer.param_groups:
            print("learning rate %.7f" % param_group["lr"])

        if opt.distributed:
            sampler_train.set_epoch(epoch)

        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)

        for i, batch_data in enumerate(train_loader):
            # the model will be evaluation mode during validation
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            cav_content = batch_data["ego"]
            batch_loss = []
            obj_index_dict = get_random_index(
                cav_content,
                opencood_train_dataset,
                1,
                mode=attack_mode,
            )
            obj_id = list(obj_index_dict.keys())[0]
            feature_index, obj_bbox = obj_index_dict[obj_id]
            attack_dict = {"attack_tagget": attack_mode}
            attack_dict["obj_idx"] = feature_index
            attack_dict["object_bbox"] = obj_bbox
            with torch.cuda.amp.autocast():
                data_dict = model.feature_encoder(cav_content)
                adv_tensor_batch = attacker.attacker.patch_apply(data_dict, attack_dict)
                output_dict = model.fusion_decoder(adv_tensor_batch)
                # output_dict = model(batch_data["ego"])
                final_loss = criterion(output_dict, batch_data["ego"]["label_dict"])

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            pbar2.update(1)

            # for name, param in model.backbone.named_parameters():
            #     if param.grad is not None:
            #         print(f"param {name} has updated")

            if not opt.half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            if hypes["lr_scheduler"]["core_method"] == "cosineannealwarm":
                scheduler.step_update(epoch * num_steps + i)

        if epoch % hypes["train_params"]["save_freq"] == 0:
            torch.save(
                model_without_ddp.state_dict(),
                os.path.join(saved_path, "net_epoch%d.pth" % (epoch + 1)),
            )

        # if epoch % hypes['train_params']['eval_freq'] == 0:
        #     valid_ave_loss = []
        #     pbar3 = tqdm.tqdm(total=len(val_loader), leave=True)
        #     print('validat at epoch %d' % (epoch))
        #     with torch.no_grad():
        #         for i, batch_data in enumerate(val_loader):
        #             model.eval()

        #             batch_data = train_utils.to_device(batch_data, device)
        #             ouput_dict = model(batch_data['ego'])

        #             final_loss = criterion(ouput_dict,
        #                                    batch_data['ego']['label_dict'])
        #             valid_ave_loss.append(final_loss.item())
        #             pbar3.update(1)
        #     valid_ave_loss = statistics.mean(valid_ave_loss)
        #     print('At epoch %d, the validation loss is %f' % (epoch,
        #                                                       valid_ave_loss))
        #     writer.add_scalar('Validate_Loss', valid_ave_loss, epoch)

    print("Training Finished, checkpoints saved to %s" % saved_path)


if __name__ == "__main__":
    main()
