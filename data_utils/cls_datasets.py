import json
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from tqdm import tqdm
from data_utils.augmentations import ToTensor
from data_utils.data_folder import ECGDatasetFolder


ts = transforms.Compose(
    [
        ToTensor(),
    ]
)


def get_data_loaders(data_path, batch_size, train_shuffle=True, weighted=False, train_ratio=1.0):
    train_dataset = ECGDatasetFolder(data_path + "train", transform=ts)
    val_dataset = ECGDatasetFolder(data_path + "val", transform=ts)
    test_dataset = ECGDatasetFolder(data_path + "test", transform=ts)

    if train_ratio < 1.0:
        train_num = len(train_dataset)
        select_train_num = int(train_num * train_ratio)
        train_dataset, _ = data.random_split(train_dataset, [select_train_num, train_num - select_train_num])

    if weighted:
        with open("class_weights.json", "r", encoding="utf-8") as fr:
            dict_weights = json.load(fr)
        class_idx = train_dataset.class_to_idx
        dict_weights_num = dict()
        for k, v in dict_weights.items():
            class_num = class_idx[k]
            dict_weights_num[class_num] = v
        print("class to idx:", class_idx)
        print("class weights: ", dict_weights_num)

        sample_weights = [0] * len(train_dataset)
        print("assign weights ....")
        for idx, (image, label) in tqdm(enumerate(train_dataset)):
            class_weight = dict_weights_num[label]
            sample_weights[idx] = class_weight
        print("Done.")
        sampler = data.WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset))
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=6)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

    return train_loader, val_loader, test_loader


class LinearClsDataset(data.Dataset):
    def __init__(self, data_path, label_path):
        self.Data = np.load(data_path)
        self.Label = np.load(label_path)

    def __len__(self):
        return len(self.Data)

    def __getitem__(self, idx):
        feat = torch.from_numpy(self.Data[idx])
        lb = torch.tensor(self.Label[idx])
        return feat.float(), lb.long()

