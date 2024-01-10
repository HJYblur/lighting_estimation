# Import necessary libraries
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model import VIDITmodel
from data_loader import VIDIT_train_dataset, VIDIT_valid_dataset, DEBUG
from evaluate import draw_loss

USE_GPU = False

def train(model, epochs):
    
    dataloader = VIDIT_train_dataset
    validloader = VIDIT_valid_dataset
    init_learning_rate = 0.01 if DEBUG else 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, factor = 0.1)
    criterion = nn.CrossEntropyLoss() 

    # model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch} training start.")
        running_loss = 0
        
        for imgs, t_labels, d_labels in tqdm(dataloader):
            if USE_GPU:
                imgs = imgs.to(device)
                t_labels = t_labels.to(device)
                d_labels = d_labels.to(device)
            optimizer.zero_grad()
            
            temp_pred, direction_pred = model(imgs)
            # print(temp_pred.size(), t_labels.size())
            temp_loss = criterion(temp_pred, t_labels)
            direction_loss = criterion(direction_pred, d_labels)
            alpha = 0.5 
            combined_loss = alpha * temp_loss + (1 - alpha) * direction_loss
            
            combined_loss.backward()
            optimizer.step()
            running_loss += combined_loss.item()
            
        else:
            valid_loss = 0
            with torch.no_grad():
                model.eval()
                
                for imgs, t_labels, d_labels in validloader:
                    if USE_GPU:
                        imgs = imgs.to(device)
                        t_labels = t_labels.to(device)
                        d_labels = d_labels.to(device)
                    t_ps, d_ps = model(imgs)
                    valid_loss += alpha * criterion(t_ps, t_labels) + (1 - alpha) * criterion(d_ps, d_labels)
                    
            model.train()
            avg_train_loss = running_loss/len(dataloader)
            avg_valid_loss = valid_loss/len(validloader)
            train_losses.append(avg_train_loss)
            valid_losses.append(avg_valid_loss)
            print(f"Current train loss: {avg_train_loss}, valid loss: {avg_valid_loss}")
            
        scheduler.step(avg_valid_loss)
            
        if epoch % 2 == 0:
            model_path = os.path.join(output_dir, f"model_{epoch}.pth")
            torch.save(model, model_path)
            
        
if torch.cuda.is_available():
    USE_GPU = True
    print("CUDA is available. Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_losses, valid_losses = [], []
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "checkpoint")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

model = VIDITmodel
if USE_GPU:
    model = model.to(device)
train_epoch = 13 if DEBUG else 50
train(model, train_epoch)
draw_loss(train_epoch, train_losses, valid_losses)