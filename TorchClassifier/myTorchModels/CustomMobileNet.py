import torch
import torch.nn as nn
from collections import namedtuple
import torchvision
from torchvision import models, transforms

# Total Parameters of this model: 3.2 million
# class DepthwiseSeparableConv(nn.Module):
#     def __init__(self, in_channels, out_channels, stride):
#         super(DepthwiseSeparableConv, self).__init__()
        
#         self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
#         self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU(inplace=True)
    
#     def forward(self, x):
#         x = self.depthwise(x)
#         x = self.pointwise(x)
#         x = self.bn(x)
#         return self.relu(x)

# class MobileNet(nn.Module):
#     def __init__(self, num_classes=1000):
#         super(MobileNet, self).__init__()
        
#         layers = [
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
            
#             DepthwiseSeparableConv(32, 64, stride=1),
#             DepthwiseSeparableConv(64, 128, stride=2),
#             DepthwiseSeparableConv(128, 128, stride=1),
#             DepthwiseSeparableConv(128, 256, stride=2),
#             DepthwiseSeparableConv(256, 256, stride=1),
#             DepthwiseSeparableConv(256, 512, stride=2),
            
#             DepthwiseSeparableConv(512, 512, stride=1),
#             DepthwiseSeparableConv(512, 512, stride=1),
#             DepthwiseSeparableConv(512, 512, stride=1),
#             DepthwiseSeparableConv(512, 512, stride=1),
#             DepthwiseSeparableConv(512, 512, stride=1),
            
#             DepthwiseSeparableConv(512, 1024, stride=2),
#             DepthwiseSeparableConv(1024, 1024, stride=1),
            
#             nn.AdaptiveAvgPool2d(1),
#             nn.Flatten(),
#             nn.Linear(1024, num_classes)
#         ]
        
#         self.model = nn.Sequential(*layers)
    
#     def forward(self, x):
#         return self.model(x)

# def count_parameters(model):
#     return sum(p.numel() for p in model.parameters() if p.requires_grad)

# def setupCustomMobileNet(numclasses):
#     model = MobileNet(numclasses)
#     print(f'The model has {count_parameters(model):,} trainable parameters')
#     return model

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)

class MobileNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNet, self).__init__()
        
        # Reduced architecture for simplicity
        layers = [
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            
            DepthwiseSeparableConv(16, 32, stride=1),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            DepthwiseSeparableConv(128, 256, stride=2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setupCustomMobileNet(numclasses):
    # Initialize our custom MobileNetV1 model.
    model = MobileNet(numclasses)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    return model
