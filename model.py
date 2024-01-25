import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


def unfreeze_model_layers(model, unfreeze_layers, print_name=False):
    if print_name:
        print("Initial state:")
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
    if unfreeze_layers != 0:
        # print(list(model.parameters()))
        for layer in list(model.parameters())[-unfreeze_layers:]:
            layer.requires_grad = True
    if print_name:
        print("\nAfter unfreezing:")
        for name, param in model.named_parameters():
            print(name, param.requires_grad)


class CustomViT(nn.Module):
    def __init__(self):
        super(CustomViT, self).__init__()
        weights = ViT_B_16_Weights.DEFAULT
        self.base_model = vit_b_16(weights=weights)  # 加载预训练模型
        for param in self.base_model.parameters():
            param.requires_grad = False

        # print(self.base_model)

        in_features = self.base_model.heads.head.in_features
        self.base_model.heads = nn.Identity()

        # 创建两个新的分类头
        self.temperature_head = nn.Linear(in_features, 5)  # 光温分类头
        self.direction_head = nn.Linear(in_features, 8)  # 光照方向分类头

    def forward(self, x):
        x = self.base_model(x)
        # 分别对两个分类头进行预测
        temp_pred = self.temperature_head(x)
        # max_temp_pred, _ = torch.max(temp_pred, 1)
        direction_pred = self.direction_head(x)
        # max_direction_pred, _ = torch.max(direction_pred, 1)
        return temp_pred, direction_pred  # 返回两个分类头的预测结果


VIDITmodel = CustomViT()
unfreeze_model_layers(VIDITmodel, 42)
