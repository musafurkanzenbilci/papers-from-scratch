import torch
import torch.nn as nn
from core.model_registry import ModelRegistry


class DepthwiseSeperableConv(nn.Module):
    def __init__(self, in_features, out_features, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            # DepthWise Conv by groups=in_features
            nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features, stride=stride),
            nn.BatchNorm2d(in_features),
            nn.ReLU()
        ) 

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


@ModelRegistry.register('mobilenet')
class MobileNet(nn.Module):
    def __init__(self, width_multiplier=0.25):
        super().__init__()
        self.alpha = width_multiplier
        depthwise_config_out_stride = [
            (64, 1),
            (128, 2),
            (128, 1),
            (256, 2),
            (256, 1),
            (512, 2),
            
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),
            (512, 1),

            (1024, 2),
            (1024, 1)
        ]

        current = int(self.alpha * 32)
        self.initial_layer = nn.Sequential(
            nn.Conv2d(3, current, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(current),
            nn.ReLU(),
        )

        depthwise_layer_list = []
        for out, stride in depthwise_config_out_stride:
            out_channels = int(out*self.alpha)
            depthwise_layer_list.append(DepthwiseSeperableConv(current, out_channels, stride=stride))
            current = out_channels

        self.depth_wise_portion = nn.Sequential(
            *depthwise_layer_list
            # DepthwiseSeperableConv(32, 64),
            # DepthwiseSeperableConv(64, 128, stride=2),
            # DepthwiseSeperableConv(128, 128),
            # DepthwiseSeperableConv(128, 256, stride=2),
            # DepthwiseSeperableConv(256, 256),
            # DepthwiseSeperableConv(256, 512, stride=2),

            # DepthwiseSeperableConv(512, 512),
            # DepthwiseSeperableConv(512, 512),
            # DepthwiseSeperableConv(512, 512),
            # DepthwiseSeperableConv(512, 512),
            # DepthwiseSeperableConv(512, 512),

            # DepthwiseSeperableConv(512, 1024, stride=2),
            # DepthwiseSeperableConv(1024, 1024),
        )

        self.avg_pool = nn.AvgPool2d(7)

        self.fc = nn.Linear(current, 10)
    
    def forward(self, x):
        x = self.initial_layer(x)
        x = self.depth_wise_portion(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
        