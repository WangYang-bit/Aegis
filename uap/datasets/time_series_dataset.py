import os
import time
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import train_utils


class TimeSeriesDataset(Dataset):
    def __init__(self, params, train=True):
        self.params = params
        self.train = train
        self.condition_frames = params["condition_frames"]
        if self.train:
            root_dir = os.path.join(params["data_dir"], params["train_dir"])
        else:
            root_dir = os.path.join(params["data_dir"], params["test_dir"])

        self.len_record = []
        self.scenario_database = OrderedDict()
        scenario_folders = sorted(
            [
                os.path.join(root_dir, x)
                for x in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, x))
            ]
        )

        for i, scenario_folder in enumerate(scenario_folders):
            self.scenario_database.update({i: OrderedDict()})
            timestamps = sorted(
                [x for x in os.listdir(scenario_folder)],
                key=lambda p: int(os.path.splitext(p)[0]),
            )
            assert len(timestamps) > 0
            for j, timestamp in enumerate(timestamps):
                self.scenario_database[i][j] = os.path.join(scenario_folder, timestamp)

            self.len_record.append(
                len(timestamps) + self.len_record[-1] - self.condition_frames
                if i > 0
                else len(timestamps) - self.condition_frames
            )

    def __getitem__(self, idx):
        scenario_index = 0
        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        scenario_database = self.scenario_database[scenario_index]
        # check the timestamp index
        timestamp_index = (
            idx + self.condition_frames
            if scenario_index == 0
            else idx - self.len_record[scenario_index - 1] + self.condition_frames
        )
        # load the data
        data = {}
        # load self.condition_frames data before the current timestamp and the current timestamp data
        feature_data = []
        idx_list = []
        for i in range(
            max((timestamp_index - self.condition_frames), 0), timestamp_index + 1
        ):
            idx_list.append(i)
            data_path = scenario_database[i]
            frame_data = torch.load(data_path)
            if i == timestamp_index:
                data["label"] = frame_data["label_dict"]
                data["scene_id"] = frame_data["scene_id"]
                data["target_feature"] = frame_data["fusion_feature"]
                break
            feature_data.append(frame_data["fusion_feature"])

        data["history_feature"] = torch.stack(feature_data, dim=0)
        data["idx_list"] = idx_list
        data["scenario_index"] = scenario_index
        return data

    def __len__(self):
        return self.len_record[-1]

    def collate_fn(self, batch):
        output = {}
        fusion_feature_list = []
        label_dict_list = []
        target_feature_list = []
        idx_lists = []
        scenario_index_list = []
        for i in range(len(batch)):
            fusion_feature_list.append(batch[i]["history_feature"])
            target_feature_list.append(batch[i]["target_feature"])
            label_dict_list.append(batch[i]["label"])
            idx_lists.append(batch[i]["idx_list"])
            scenario_index_list.append(batch[i]["scenario_index"])
        label_torch_dict = self.collate_batch(label_dict_list)
        output.update(
            {
                "history_feature": torch.stack(fusion_feature_list, dim=0),
                "target_feature": torch.stack(target_feature_list, dim=0),
                "label_dict": label_torch_dict,
                "idx_list": idx_lists,
                "scenario_index": scenario_index_list,
            }
        )
        return output

    @staticmethod
    def collate_batch(label_batch_list):
        """
        Customized collate function for target label generation.

        Parameters
        ----------
        label_batch_list : list
            The list of dictionary  that contains all labels for several
            frames.

        Returns
        -------
        target_batch : dict
            Reformatted labels in torch tensor.
        """
        pos_equal_one = []
        neg_equal_one = []
        targets = []

        for i in range(len(label_batch_list)):
            pos_equal_one.append(label_batch_list[i]["pos_equal_one"])
            neg_equal_one.append(label_batch_list[i]["neg_equal_one"])
            targets.append(label_batch_list[i]["targets"])

        pos_equal_one = torch.stack(pos_equal_one, dim=0)
        neg_equal_one = torch.stack(neg_equal_one, dim=0)
        targets = torch.stack(targets, dim=0)

        return {
            "targets": targets,
            "pos_equal_one": pos_equal_one,
            "neg_equal_one": neg_equal_one,
        }


if __name__ == "__main__":
    params = {
        "root_dir": "/workspace/AdvCollaborativePerception/uap/data/fusion_feature/train",
        "validate_dir": "path/to/validate/dir",
        "condition_frames": 10,
        "cache_size": 100,
    }
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    dataset = TimeSeriesDataset(params, train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        num_workers=16,
        collate_fn=dataset.collate_fn,
        shuffle=True,
    )
    start = time.time()
    num_steps = len(dataset)
    # pbar2 = tqdm(total=num_steps, leave=True)
    for i, batch_data in tqdm(enumerate(train_loader)):
        batch_data = train_utils.to_device(batch_data, device)
        print(batch_data)
        continue
    end = time.time()
    print(i)
