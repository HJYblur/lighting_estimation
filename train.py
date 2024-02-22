import os
import torch
import torch.nn as nn
from tqdm import tqdm
import tkinter as tk
from resnet18 import ResNet18
from data_loader import VIDIT_train_loader, VIDIT_valid_loader, DEBUG, aug_size
from utils import draw_loss

USE_GPU = True


# 单选框，选择当前训练的是学习光温还是光照方向的模型
def show_radio_selection():
    root = tk.Tk()
    var = tk.IntVar()
    # 创建单选框
    t_radio = tk.Radiobutton(root, text="t", variable=var, value=1)
    d_radio = tk.Radiobutton(root, text="d", variable=var, value=2)

    # 创建按钮用于获取选择
    submit_button = tk.Button(root, text="确定", command=root.destroy)  # 关闭窗口

    # 布局
    t_radio.pack()
    d_radio.pack()
    submit_button.pack()

    # 启动主循环
    root.mainloop()

    return var.get()


def to_cpu_list(tensor_list):
    return [tensor.item() for tensor in tensor_list]


def train(model, epochs, mode, train_losses, valid_losses):
    init_learning_rate = 1e-3
    optimizer = torch.optim.Adam(
        model.parameters(), lr=init_learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    # model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch} training start.")
        model.train()
        running_loss = 0

        for data in tqdm(VIDIT_train_loader):
            for imgs, t_label, d_label in data:
                label = t_label if mode == "t" else d_label
                if USE_GPU:
                    imgs = imgs.to(device)
                    label = label.to(device)

                optimizer.zero_grad()

                pred = model(imgs)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        else:
            valid_loss = 0
            with torch.no_grad():
                model.eval()

                for data in VIDIT_valid_loader:
                    for imgs, t_label, d_label in data:
                        label = t_label if mode == "t" else d_label
                        if USE_GPU:
                            imgs = imgs.to(device)
                            label = label.to(device)

                        pred = model(imgs)
                        valid_loss += criterion(pred, label)

            avg_train_loss = running_loss / (len(VIDIT_train_loader) * aug_size)
            avg_valid_loss = valid_loss / (len(VIDIT_valid_loader) * aug_size)

            train_losses.append(avg_train_loss)
            valid_losses.append(avg_valid_loss)

            print(f"Current train loss: {avg_train_loss}, valid loss: {avg_valid_loss}")

        scheduler.step(avg_valid_loss)

        # 每隔两轮存储一次训练好的模型
        if epoch % 2 == 0:
            model_path = os.path.join(output_dir, mode, f"model_{epoch}.pth")
            torch.save(model, model_path)


if __name__ == "__main__":
    if torch.cuda.is_available():
        USE_GPU = True
        print("CUDA is available. Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_channels = 3
    train_epoch = 2 if DEBUG else 50
    current_dir = os.getcwd()
    output_dir = os.path.join(current_dir, "checkpoint")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 分别建立一个分类头是8和5的模型
    dir_model = ResNet18(image_channels, 8)
    temp_model = ResNet18(image_channels, 5)
    if USE_GPU:
        dir_model = dir_model.to(device)
        temp_model = temp_model.to(device)

    train_dir_losses, train_temp_losses, valid_dir_losses, valid_temp_losses = (
        [],
        [],
        [],
        [],
    )
    choice = show_radio_selection()
    train(dir_model, train_epoch, choice, train_dir_losses, valid_dir_losses)

    # if (
    #     len(train_losses) > 0
    #     and isinstance(train_losses[0], torch.Tensor)
    #     and train_losses[0].is_cuda
    # ):
    #     train_losses = to_cpu_list(train_losses)
    train_dir_losses = to_cpu_list(train_dir_losses)
    valid_dir_losses = to_cpu_list(valid_dir_losses)
    draw_loss(
        train_epoch,
        train_dir_losses,
        valid_dir_losses,
        "Direction's Training and Validation Loss Over Epochs",  # "Direction's Training and Validation Loss Over Epochs",
    )

    # train_temp_losses = to_cpu_list(train_temp_losses)
    # valid_temp_losses = to_cpu_list(valid_temp_losses)
    # draw_loss(train_epoch, train_temp_losses, valid_temp_losses, "Temperature's Training and Validation Loss Over Epochs")
