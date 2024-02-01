import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog
from PIL import Image
from test import load_model, load_answer
from data_loader import data_transform, TEMP_LIST, DIR_LIST


def preprocess_img(file_path):
    image = Image.open(file_path).convert("RGB")
    img = data_transform(image).unsqueeze(0)
    return img.to(device)

def load_answer(img, model, mode):
    with torch.no_grad():
        pred = model(img)
        _, indices = torch.max(pred, dim=1)
        val = indices.item()
        return TEMP_LIST[val] if mode =='t' else DIR_LIST[val]

if __name__ == "__main__":
    if torch.cuda.is_available():
        USE_GPU = True
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    print(f"当前选择的文件路径：{file_path}")
    
    target_image = preprocess_img(file_path)

    # 计算当前光照方向
    dir_model = load_model(8, "d")
    direction = load_answer(target_image, dir_model, 'd')
    
    # 计算光温
    temp_model = load_model(8, 't')
    temperature = load_answer(target_image, temp_model, 't')

    # 计算光照强度和天光
    intensity, sky_light = analyze_brightness(file_path)
    
    # 存入CSV文件等待与UE进行交互
    print(f"direction: {direction},\ntemperature: {temperature},\nintensity: {intensity},\nsky_light:{sky_light}")
