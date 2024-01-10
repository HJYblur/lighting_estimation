import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

DEBUG = True

DATA_PATH = '../Data'
TRAIN_PATH = DATA_PATH + ('/train_s/' if DEBUG else '/train') 
VALID_PATH = DATA_PATH + ('/valid_s/' if DEBUG else '/valid')
TEST_PATH = DATA_PATH + '/test/'

TEMP_LIST = ["2500", "3500", "4500", "5500", "6500"]
temp_to_idx = {temp: idx for idx, temp in enumerate(TEMP_LIST)}
DIR_LIST = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
dir_to_index = {dir: idx for idx, dir in enumerate(DIR_LIST)}


def load_temp_label(item):
    temp = item.temperature
    return torch.tensor(temp_to_idx[temp], dtype = torch.long)

def load_dirc_label(item):
    dirc = item.direction
    return torch.tensor(dir_to_index[dirc], dtype = torch.long)


class VIDITData:
    def __init__(self, data_dir):
        self.name = data_dir
        self.id = self.name.split("_")[0]
        self.temperature = self.name.split("_")[1]
        self.direction = self.name.split("_")[2].split(".")[0]

class VIDITDataset(Dataset): 
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)  # 获取文件列表

    def __getitem__(self, idx):
        filename = self.file_list[idx]  # 使用索引获取文件名
        item = VIDITData(filename)
        # img =  open_pic(self.data_dir, item)
        img = VIDITtransform(Image.open(os.path.join(self.data_dir, filename)).convert('RGB'))
        temp_label = load_temp_label(item)
        dirc_label = load_dirc_label(item)
        # print(img.size())
        return img, temp_label, dirc_label
    
    def __len__(self):
        return len(self.file_list)
        
VIDITtransform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

batch_size = 64 if DEBUG else 256
VIDIT_train_dataset = torch.utils.data.DataLoader(VIDITDataset(data_dir=TRAIN_PATH), batch_size=batch_size, shuffle=True)
VIDIT_valid_dataset = torch.utils.data.DataLoader(VIDITDataset(data_dir=VALID_PATH), batch_size=batch_size, shuffle=True)

# 测试数据读入
# if DEBUG:
#     image, t_label, d_label = next(iter(VIDIT_train_dataset))
#     imagedemo = image[3].permute(1,2,0) 
#     imagedemo = torch.clamp(imagedemo, 0, 1)
#     print(imagedemo.size())
#     t_demolabel = t_label[3]
#     d_demolabel = d_label[3]

#     plt.imshow(imagedemo)
#     plt.axis('on')  # 显示坐标轴
#     plt.show()
#     print(t_demolabel, d_demolabel)

