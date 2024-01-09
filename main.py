# Import necessary libraries
import os
import torch
import torch.nn as nn
from tqdm import tqdm
from model import CustomViT
from data_loader import VIDIT_train_dataset, VIDIT_test_dataset
from evaluate import draw_loss

train_losses, test_losses = [], []
def train(model, epochs):
    
    dataloader = VIDIT_train_dataset
    testloader = VIDIT_test_dataset
    init_learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=init_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 5, factor = 0.1)
    criterion = nn.CrossEntropyLoss() 

    # model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch} training start.")
        running_loss = 0
        
        for imgs, t_labels, d_labels in tqdm(dataloader):
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
            test_loss = 0
            t_accuracy = 0
            d_accuracy = 0
            with torch.no_grad():
                model.eval()
                
                for imgs, t_labels, d_labels in testloader:
                    t_ps, d_ps = model(imgs)
                    test_loss += alpha * criterion(t_ps, t_labels) + (1 - alpha) * criterion(d_ps, d_labels)
                    # t_top, t_class = t_ps.topk(1, dim = 1)
                    # d_top, d_class = d_ps.topk(1, dim = 1)
                    # t_equal = t_class == t_labels.view(*t_top.shape)
                    # d_equal = d_class == d_labels.view(*d_top.shape)
                    # t_accuracy += torch.mean(t_equal.type(torch.FloatTensor))
                    # d_accuracy += torch.mean(d_equal.type(torch.FloatTensor))
                    
            model.train()
            avg_train_loss = running_loss/len(dataloader)
            avg_test_loss = test_loss/len(testloader)
            train_losses.append(avg_train_loss)
            test_losses.append(avg_test_loss)
            print(f"Current train loss: {avg_train_loss}, test loss: {avg_test_loss}")
            
        scheduler.step(avg_test_loss)
            
        if epoch % 2 == 0:
            model_path = os.path.join(output_dir, f"model_{epoch}.pth")
            torch.save(model, model_path)
            
    return
        
current_dir = os.getcwd()
output_dir = os.path.join(current_dir, "checkpoint")
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
train(CustomViT(), epochs=10)
draw_loss(train_losses, test_losses)