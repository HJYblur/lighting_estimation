import os
import torch
import torch.nn as nn
from tqdm import tqdm
from resnet18 import ResNet18
from data_loader import VIDIT_train_loader, VIDIT_valid_loader, DEBUG, aug_size
from utils import draw_loss

USE_GPU = True


def to_cpu_list(tensor_list):
    return [tensor.item() for tensor in tensor_list]


def train(model, epochs, mode, train_losses, valid_losses):
    init_learning_rate = 1e-5
    optimizer = torch.optim.Adam(
        model.parameters(), lr=init_learning_rate, weight_decay=1e-5
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
    train(dir_model, train_epoch, "t", train_dir_losses, valid_dir_losses)

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
        "Temperature's Training and Validation Loss Over Epochs",
    )

    # train_temp_losses = to_cpu_list(train_temp_losses)
    # valid_temp_losses = to_cpu_list(valid_temp_losses)
    # draw_loss(train_epoch, train_temp_losses, valid_temp_losses, "Temperature's Training and Validation Loss Over Epochs")
