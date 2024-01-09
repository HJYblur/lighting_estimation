import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class ImageEncoding():
    def __init__(self, in_channels=3, enc_channels=32, out_channels=64):
        super(ImageEncoding, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, enc_channels - in_channels, 7, padding=3),
            nn.BatchNorm2d(enc_channels - in_channels),
            nn.PReLU()
         )
        self.conv2 = nn.Sequential(
            nn.Conv2d(enc_channels, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
         )
    
    def forward(self, image):
        x = self.conv1(image)
        encoded_image = torch.cat((x, image), dim=1)  # append image channels to the convolution result
        self.skip_connections = [encoded_image]
        return self.conv2(encoded_image)

class CustomViT(nn.Module):
    def __init__(self):
        super(CustomViT, self).__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.base_model = vit_b_16(weights=weights)  # 加载预训练模型
        for param in self.base_model.heads.parameters():
            param.requires_grad = False
        
        # print(self.base_model)
        
        in_features = self.base_model.heads.head.in_features
        self.base_model.heads = nn.Identity()
        
        # 创建两个新的分类头
        self.temperature_head = nn.Linear(in_features, 5)  # 光温分类头
        self.direction_head = nn.Linear(in_features, 8)    # 光照方向分类头

    def forward(self, x):
        x = self.base_model(x)
        # 分别对两个分类头进行预测
        temp_pred = self.temperature_head(x)
        # max_temp_pred, _ = torch.max(temp_pred, 1)
        direction_pred = self.direction_head(x)
        # max_direction_pred, _ = torch.max(direction_pred, 1)
        return temp_pred, direction_pred  # 返回两个分类头的预测结果
    
model = CustomViT()
