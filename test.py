import os
import torch
import random
from PIL import Image
from data_loader import VIDITData, VIDITtransform, VALID_PATH, TEMP_LIST, DIR_LIST

def load_model(num):
    # 加载模型
    model_path = os.path.join(os.getcwd(), "output", f"model_{num}.pth")
    VIT_model = torch.load(model_path)
    VIT_model.eval()
    return VIT_model


def preprocess_img(img_name):
    image = Image.open(os.path.join(VALID_PATH, img_name)).convert('RGB')
    img = VIDITtransform(image).unsqueeze(0)
    return img


def load_pic():
    # 加载测试图片
    files = [f for f in os.listdir(VALID_PATH)]
    img_name = random.choice(files) if files else None
    # print(img_name)
    img = preprocess_img(img_name)
    return img, img_name


def load_batch_pic(num):
    for n in range(num):
        img, name = load_pic()
        img_list.append(img)
        name_list.append(name)
    return img_list, name_list

def load_answer(img):
    with torch.no_grad():
        temp_pred, direction_pred = VIDITmodel(img)
        _, temp_indices = torch.max(temp_pred, dim = 1)
        _, dir_indices = torch.max(direction_pred, dim = 1)
        temp_val = temp_indices.item()
        dir_val = dir_indices.item()
        return TEMP_LIST[temp_val], DIR_LIST[dir_val]
    
def batch_test(epoch):
    t_accuracy = 0
    d_accuracy = 0
    load_batch_pic(epoch)
    for e in range(epoch):
        name = name_list[e]
        item = VIDITData(name)
        
        img = img_list[e]
        t_pred, d_pred = load_answer(img)
        
        t_accuracy += t_pred == item.temperature
        d_accuracy += d_pred == item.direction
    print(f"{epoch} epoch's accuracy:\nTemperature: {t_accuracy/epoch}, Direction:{d_accuracy/epoch}")
    return

img_list = []
name_list = []
VIDITmodel = load_model(8)
batch_test(20)