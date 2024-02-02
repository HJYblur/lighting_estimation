import os
import torch
import random
from PIL import Image
from tqdm import tqdm
from data_loader import (
    CustomData,
    data_transform,
    TRAIN_PATH,
    TEST_PATH,
    TEMP_LIST,
    DIR_LIST,
)


def load_model(num, mode):
    # 加载模型
    model_path = os.path.join(os.getcwd(), "checkpoint", mode, f"model_{num}.pth")
    model = torch.load(model_path)
    model.eval()
    return model.to(device)


def preprocess_img(img_name):
    image = Image.open(os.path.join(TEST_PATH, img_name)).convert("RGB")
    # image.show()
    img = data_transform(image).unsqueeze(0)
    return img.to(device)


def load_pic():
    # 加载测试图片
    files = [f for f in os.listdir(TEST_PATH)]
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
        t_pred = t_model(img)
        d_pred = d_model(img)
        # print(t_pred, d_pred)
        _, temp_indices = torch.max(t_pred, dim=1)
        _, dir_indices = torch.max(d_pred, dim=1)
        temp_val = temp_indices.item()
        dir_val = dir_indices.item()
        return TEMP_LIST[temp_val], DIR_LIST[dir_val]


def batch_test(epoch):
    t_accuracy = 0
    d_accuracy = 0
    load_batch_pic(epoch)
    for e in tqdm(range(epoch)):
        name = name_list[e]
        item = CustomData(name)

        img = img_list[e]
        t_pred, d_pred = load_answer(img)

        t_accuracy += t_pred == item.temperature
        d_accuracy += d_pred == item.direction
        # print(f"当前照片：{name}, 预测结果：{t_pred}, {d_pred}")
    print(
        f"{epoch} epoch's accuracy:\nTemperature: {t_accuracy/epoch}, Direction:{d_accuracy/epoch}"
    )
    return


def sum_test():
    t_accuracy = 0
    d_accuracy = 0
    epoch = 0
    for name in tqdm(os.listdir(TEST_PATH)):
        img = preprocess_img(name)
        item = CustomData(name)
        t_pred, d_pred = load_answer(img)

        t_accuracy += t_pred == item.temperature
        d_accuracy += d_pred == item.direction
        epoch += 1
    print(
        f"{epoch} epoch's accuracy:\nTemperature: {t_accuracy/epoch}, Direction:{d_accuracy/epoch}"
    )
    return


if __name__ == "__main__":
    img_list = []
    name_list = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_model = load_model(18, "t")
    d_model = load_model(6, "d")
    batch_test(100)
    # sum_test()
