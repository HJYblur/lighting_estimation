import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEBUG = False

DATA_PATH = "../Data"
TRAIN_PATH = DATA_PATH + ("/train_s/" if DEBUG else "/train")
VALID_PATH = DATA_PATH + ("/valid_s/" if DEBUG else "/valid")
TEST_PATH = DATA_PATH + "/test/"

TEMP_LIST = ["2500", "3500", "4500", "5500", "6500"]
temp_to_idx = {temp: idx for idx, temp in enumerate(TEMP_LIST)}
DIR_LIST = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
dir_to_index = {dir: idx for idx, dir in enumerate(DIR_LIST)}


def add_noise(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    noise = np.random.randint(0, 256, image.shape, dtype="uint8")
    noise_image = cv2.add(image, noise)
    return noise_image


def load_temp_label(item):
    temp = item.temperature
    return torch.tensor(temp_to_idx[temp], dtype=torch.long)


def load_dirc_label(item):
    dirc = item.direction
    return torch.tensor(dir_to_index[dirc], dtype=torch.long)


class CustomData:
    def __init__(self, data_dir):
        self.name = data_dir
        self.id = self.name.split("_")[0]
        self.temperature = self.name.split("_")[1]
        self.direction = self.name.split("_")[2].split(".")[0]


class CustomDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)  # 获取文件列表
        self.augmentation_transform = transforms.Compose(
            [
                # transforms.RandomRotation(10),
                transforms.RandomCrop((10, 10))
            ]
        )

    def __getitem__(self, idx):
        filename = self.file_list[idx]  # 使用索引获取文件名
        img = data_transform(
            Image.open(os.path.join(self.data_dir, filename)).convert("RGB")
        )

        item = CustomData(filename)
        temp_label = load_temp_label(item)
        dirc_label = load_dirc_label(item)
        datalist = [
            (self.augmentation_transform(img), temp_label, dirc_label) for _ in range(5)
        ]
        # datalist = [
        #     (add_noise(image), t_label, d_label) for image, t_label, d_label in datalist
        # ]

        return datalist

    def __len__(self):
        return len(self.file_list)


data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor(),
        # transforms.Normalize((0.5,), (0.5,)),
    ]
)


batch_size = 32
aug_size = 8
VIDIT_train_loader = torch.utils.data.DataLoader(
    CustomDataset(data_dir=TRAIN_PATH), batch_size=batch_size, shuffle=True
)
VIDIT_valid_loader = torch.utils.data.DataLoader(
    CustomDataset(data_dir=VALID_PATH), batch_size=batch_size, shuffle=True
)
