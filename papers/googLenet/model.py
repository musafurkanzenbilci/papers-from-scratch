import torch.nn as nn
import torch
from core.model_registry import ModelRegistry

class InceptionBlockNaive(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_features, int(out_features/4), kernel_size=1)
        self.relu1x1 = nn.ReLU()
        self.conv3x3 = nn.Conv2d(in_features, int(out_features/2), kernel_size=3, padding=1)
        self.relu3x3 = nn.ReLU()
        self.conv5x5 = nn.Conv2d(in_features, int(out_features/8), kernel_size=5, padding=2)
        self.relu5x5 = nn.ReLU()

        self.pool3x3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        # Not really optimal because the output filters of MaxPool is equal ot the previous layer

    
    def forward(self, x):
        out1 = self.conv1x1(x)
        print(out1.shape)
        out2 = self.conv3x3(x)
        print(out2.shape)
        out3 = self.conv5x5(x)
        print(out3.shape)
        out4 = self.pool3x3(x)
        print(out4.shape)
        return x


class InceptionBlockDimReduction(nn.Module):
    def __init__(
            self,
            in_features,
            c1x1,
            c1x3,    
            c3x3,
            c1x5,
            c5x5,
            pool_proj,    
        ):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_features, c1x1, kernel_size=1)
        self.relu1x1 = nn.ReLU()

        self.conv1x3 = nn.Conv2d(in_features, c1x3, kernel_size=1)
        self.relu1x3 = nn.ReLU()
        self.conv3x3 = nn.Conv2d(c1x3, c3x3, kernel_size=3, padding=1)
        self.relu3x3 = nn.ReLU()

        self.conv1x5 = nn.Conv2d(in_features, c1x5, kernel_size=1)
        self.relu1x5 = nn.ReLU()
        self.conv5x5 = nn.Conv2d(c1x5, c5x5, kernel_size=5, padding=2)
        self.relu5x5 = nn.ReLU()

        self.pool3x3 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.convpoolx1 = nn.Conv2d(in_features, pool_proj, kernel_size=1)
        self.relupoolx1 = nn.ReLU()

    def forward(self, x):
        branch1 = self.conv1x1(x)
        branch1 = self.relu1x1(branch1)

        branch2 = self.conv1x3(x)
        branch2 = self.relu1x3(branch2)
        branch2 = self.conv3x3(branch2)
        branch2 = self.relu3x3(branch2)
        
        branch3 = self.conv1x5(x)
        branch3 = self.relu1x5(branch3)
        branch3 = self.conv5x5(branch3)
        branch3 = self.relu5x5(branch3)

        branch4 = self.pool3x3(x)
        branch4 = self.convpoolx1(branch4)
        branch4 = self.relupoolx1(branch4)

        x = torch.cat([branch1, branch2, branch3, branch4], dim=1)

        return x


@ModelRegistry.register('googlenet')
class GoogleNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.initial_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.inception_blocks = nn.Sequential(
            InceptionBlockDimReduction(in_features=192, c1x1=64, c1x3=96, c3x3=128, c1x5=16, c5x5=32, pool_proj=32),
            InceptionBlockDimReduction(in_features=256, c1x1=128, c1x3=128, c3x3=192, c1x5=32, c5x5=96, pool_proj=64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlockDimReduction(in_features=480, c1x1=192, c1x3=96, c3x3=208, c1x5=16, c5x5=48, pool_proj=64),
            InceptionBlockDimReduction(in_features=512, c1x1=160, c1x3=112, c3x3=224, c1x5=24, c5x5=64, pool_proj=64),
            InceptionBlockDimReduction(in_features=512, c1x1=128, c1x3=128, c3x3=256, c1x5=24, c5x5=64, pool_proj=64),
            InceptionBlockDimReduction(in_features=512, c1x1=112, c1x3=144, c3x3=288, c1x5=32, c5x5=64, pool_proj=64),
            InceptionBlockDimReduction(in_features=528, c1x1=256, c1x3=160, c3x3=320, c1x5=32, c5x5=128, pool_proj=128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            InceptionBlockDimReduction(in_features=832, c1x1=256, c1x3=160, c3x3=320, c1x5=32, c5x5=128, pool_proj=128),
            InceptionBlockDimReduction(in_features=832, c1x1=384, c1x3=192, c3x3=384, c1x5=48, c5x5=128, pool_proj=128),
            nn.AvgPool2d(kernel_size=7, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(1024, 10)
        )
    
    def forward(self, x):
        x = self.initial_features(x)
        x = self.inception_blocks(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x