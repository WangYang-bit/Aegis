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
from models.ar import FeatureTimeSeriesTransformer


class AR_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.pillar_loss = PointPillarLoss({"cls_weight": 1.0, "reg": 2.0})
        self.mse_loss = nn.MSELoss()
        self.loss_dict = {}

    def forward(self, output, target):
        pillar_loss = self.pillar_loss(output, target["label_dict"])
        mse_loss = self.mse_loss(output["pred_features"], target["target_feature"])
        total_loss = self.alpha * mse_loss + self.beta * pillar_loss

        self.loss_dict.update(
            {
                "total_loss": total_loss,
                "pillar_loss": pillar_loss,
                "mse_loss": mse_loss,
            }
        )
        return total_loss

    def logging(self, epoch, batch_id, batch_len, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict["total_loss"]
        pillar_loss = self.loss_dict["pillar_loss"]
        mse_loss = self.loss_dict["mse_loss"]
        if pbar is None:
            print(
                "[epoch %d][%d/%d], || Loss: %.4f || Conf Loss: %.4f"
                " || Loc Loss: %.4f"
                % (
                    epoch,
                    batch_id + 1,
                    batch_len,
                    total_loss.item(),
                    pillar_loss.item(),
                    mse_loss.item(),
                )
            )
        else:
            pbar.set_description(
                "[epoch %d][%d/%d], || Loss: %.4f || Pillar Loss: %.4f"
                " || Mse Loss: %.4f"
                % (
                    epoch,
                    batch_id + 1,
                    batch_len,
                    total_loss.item(),
                    pillar_loss.item(),
                    mse_loss.item(),
                )
            )


class WorldModel(nn.Module):
    def __init__(self, args, device):
        super(WorldModel, self).__init__()
        self.args = args
        self.condition_frames = args["condition_frames"]
        self.total_token_size = args["total_token_size"]
        self.feature_token_size = args["feature_token_size"]
        self.pose_token_size = args["pose_token_size"]
        self.yaw_token_size = args["yaw_token_size"]
        self.feature_size = args["feature_size"]
        self.downsample_factor = args["downsample_factor"]
        self.latent_size = [int(x / self.downsample_factor) for x in self.feature_size]
        self.input_dim = args["input_dim"]

        self.token_size_dict = {
            "img_tokens_size": self.feature_token_size,
            "pose_tokens_size": self.pose_token_size,
            "yaw_token_size": self.yaw_token_size,
            "total_tokens_size": self.total_token_size,
        }
        transformer_config = {
            "block_size": self.condition_frames * (self.total_token_size),
            "n_layer": args["n_layer"],
            "n_head": args["n_head"],
            "n_embd": args["embedding_dim"],
            "condition_frames": self.condition_frames,
            "token_size_dict": self.token_size_dict,
            "latent_size": self.latent_size,
            "L": self.feature_token_size,
            "device": device,
        }

        self.backbone = FeatureTimeSeriesTransformer(**transformer_config)

        self.downsample = Downsample(args["input_dim"], args["embedding_dim"])

        self.upsample = Upsample(args["embedding_dim"], args["input_dim"])

        self.cls_head = nn.Conv2d(
            args["input_dim"], args["anchor_number"], kernel_size=1
        )
        self.reg_head = nn.Conv2d(
            args["input_dim"], 7 * args["anchor_number"], kernel_size=1
        )

    def encode(self, inputs):
        # inputs: (B, T, C, H, W)
        B, T, C, H, W = inputs.size()
        assert C == self.input_dim, (
            f"Expected input feature dimension {self.input_dim}, but got {C}"
        )

        x = inputs.view(-1, *inputs.shape[2:])  # (B*T, C, H, W)
        # (B*T, C, H, W) -> (B*T, d_model, H/2, W/2)
        x = self.downsample(x)
        # (B*T, d_model, H/2, W/2) -> (B, T, H*W/4, d_model)
        x = rearrange(x, "(b f) c h w -> b f (h w) c", b=B)
        # (B, T, H*W/4, d_model) -> (B, H*W/4, embeding_dim)
        x = self.backbone(x)
        return x

    def decode(self, x):
        B, _, C = x.size()
        x = rearrange(
            x,
            "b  (h w) c -> b c h w",
            b=B,
            h=self.latent_size[0],
            w=self.latent_size[1],
        )
        # ( B,  embeding_dim, H/2, W/2) to (B, feature_dim, H, W )
        x = self.upsample(x)

        psm = self.cls_head(x)
        rm = self.reg_head(x)

        return {
            "pred_features": x,
            "rm": rm,
            "psm": psm,
        }

    def forward(self, inputs):
        x = inputs["history_feature"]
        x = self.encode(x)
        output = self.decode(x)
        return output

class Upsample(nn.Module):
    def __init__(self, in_channels=1024, out_channels=256):
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels

        self.convs = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=512,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                in_channels=512,
                out_channels=self.out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                output_padding=0,
            ),
            # nn.ReLU(inplace=True),
            # nn.ConvTranspose2d(512, out_dim, 3, padding=1),
        )

    def forward(self, x):
        return self.convs(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.convs = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=512,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=512,
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, latent_dim, kernel_size=2, stride=2, padding=0),
        )

    def forward(self, x):
        return self.convs(x)


if __name__ == "__main__":
    config_file = "/workspace/AdvCollaborativePerception/uap/configs/uap.yaml"
    config = yaml_utils.load_yaml(config_file, None)
    args = config["model"]["world_model"]
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    dataset_params = config["data"]

    print("-----------------Dataset Building------------------")
    train_dataset = TimeSeriesDataset(dataset_params, train=True)
    val_dataset = TimeSeriesDataset(dataset_params, train=False)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        num_workers=16,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=2,
        num_workers=16,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="uap",
        # track hyperparameters and run metadata
        config={
            "dataset": "Timeseries_Feature",
            "epochs": 20,
        },
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

    model.to(device)
    criterion = AR_Loss(
        alpha=config["AR_loss"]["args"]["mse_weight"],
        beta=config["AR_loss"]["args"]["task_weight"],
    )

    # optimizer setup
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )

    start = time.time()
    for epoch in range(
        init_epoch + 1, max(config["train_params"]["max_epoch"], init_epoch + 1)
    ):
        pbar2 = tqdm(total=len(train_loader), leave=True)
        best_loss = 100000

        for i, batch_data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            target = {
                "target_feature": batch_data["target_feature"],
                "label_dict": batch_data["label_dict"],
            }
            output = model(batch_data)  # (B, H, W, C)

            loss = criterion(output, target)
            criterion.logging(epoch, i, len(train_loader), pbar=pbar2)
            if loss < best_loss:
                best_loss = loss
            # log metrics to wandb
            wandb.log(
                {
                    "loss": loss,
                    "mes_loss": criterion.loss_dict["mse_loss"],
                    "pillar_loss": criterion.loss_dict["pillar_loss"],
                    "best_loss": best_loss,
                }
            )
            loss.backward()
            optimizer.step()
            pbar2.update(1)

        if epoch % config["train_params"]["save_interval"] == 0:
            torch.save(
                model.state_dict(),
                f"/workspace/AdvCollaborativePerception/uap/models/ar_model/net_epoch{epoch}.pth",
            )

        if epoch % config["train_params"]["eval_freq"] == 0:
            valid_ave_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    input_tensor = batch_data["history_feature"]
                    target = {
                        "target_feature": batch_data["target_feature"],
                        "label_dict": batch_data["label_dict"],
                    }
                    output = model(input_tensor)

                    final_loss = criterion(output, target)
                    valid_ave_loss.append(final_loss.item())
            print(
                "At epoch %d, the validation loss is %f"
                % (epoch, np.mean(valid_ave_loss))
            )
            wandb.log({"Validate_Loss": np.mean(valid_ave_loss), "epoch": epoch})
    wandb.finish()
