import os
import json
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from test import load_model, load_answer
from data_loader import data_transform, TEMP_LIST, DIR_LIST
from calculation import analyze_brightness
from utils import direction_vectors


def is_dir_empty(dir_path):
    return not bool(os.listdir(dir_path))

def preprocess_img(file_path):
    image = Image.open(file_path).convert("RGB")
    img = data_transform(image).unsqueeze(0)
    return img.to(device)


def load_model(mode):
    # 加载模型
    model_path = default_temp_model_path if mode == 't' else default_dir_model_path
    model = torch.load(model_path)
    model.eval()
    return model.to(device)

def load_answer(img, model, mode):
    with torch.no_grad():
        pred = model(img)
        _, indices = torch.max(pred, dim=1)
        val = indices.item()
        return TEMP_LIST[val] if mode =='t' else DIR_LIST[val]

if __name__ == "__main__":
    if is_dir_empty("./pre_trained"):
        print("当前无预训练模型，请先进行训练。")
    
    if torch.cuda.is_available():
        USE_GPU = True
        print("CUDA is available. Working on GPU.")
    else:
        print("CUDA is not available. Working on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    print(f"当前选择的文件路径：{file_path}")
    
    target_image = preprocess_img(file_path)
    
    default_temp_model_path = os.path.join(os.getcwd(), "pre_trained", "temp20.pth")
    default_dir_model_path = os.path.join(os.getcwd(), "pre_trained", "direc6.pth")

    # 计算当前光照方向
    dir_model = load_model("d")
    direction = load_answer(target_image, dir_model, 'd')
    direction_vec = direction_vectors[direction]
    
    # 计算光温
    temp_model = load_model('t')
    temperature = load_answer(target_image, temp_model, 't')

    # 计算光照强度和天光
    intensity, sky_light = analyze_brightness(file_path)
    
    print(f"direction: {direction},\ntemperature: {temperature},\nintensity: {intensity},\nsky_light:{sky_light}")
    
    # 存入CSV文件等待与UE进行交互
    data = {
        "name": file_path.split("/")[-1],
        "dimention0": direction_vec[0],
        "dimention1": direction_vec[1],
        "temperature": temperature,
        "intensity": intensity,
        "sky_light": sky_light
    }
    with open("./output/answer.json", 'w') as file:
        json.dump(data, file, indent=6)
        print("文件已存储到D:\File_Lemon\lighting tagger\lighting_estimation\output\\answer.json")
