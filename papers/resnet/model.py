import torch.nn as nn
import torch
from core.model_registry import ModelRegistry


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, filter=3, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_dim, out_dim, filter, padding=1, stride=stride)
        self.bnorm1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_dim, out_dim, filter, padding=1, stride=1)
        self.bnorm2 = nn.BatchNorm2d(out_dim)

        # If no change in dim or stride, input stay the same
        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_dim)
            )

        self.relu2 = nn.ReLU()
        

    def forward(self, x):
        shortcut_x = self.shortcut(x) 

        x = self.conv1(x)
        x = self.bnorm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bnorm2(x)

        x = torch.add(shortcut_x, x)
        x = self.relu2(x)

        return x


@ModelRegistry.register('resnet')
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
         
        self.features = nn.Sequential(
            ResBlock(in_dim=64, out_dim=64),
            ResBlock(in_dim=64, out_dim=64),
            
            ResBlock(in_dim=64, out_dim=128, stride=2),
            ResBlock(in_dim=128, out_dim=128),

            ResBlock(in_dim=128, out_dim=256, stride=2),
            ResBlock(in_dim=256, out_dim=256),

            ResBlock(in_dim=256, out_dim=512, stride=2),
            ResBlock(in_dim=512, out_dim=512),   
        )

        self.final_pool = nn.AdaptiveAvgPool2d((1, 1))        
        self.fc = nn.Linear(512, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)

        x = self.features(x)

        x = self.final_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x
