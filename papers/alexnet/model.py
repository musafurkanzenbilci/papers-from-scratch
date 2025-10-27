import torch
import torch.nn as nn
from core.model_registry import ModelRegistry


# region Original AlexNet Architecture for ImageNet with 256x256 images
# --------data preprocess----------------
# Image Translation from 256x256 images to 224x224 patches and their reflections
# Perform PCA on the set of RGB values
#----------------------------------------
# Conv1(kernels=96, filter=11x11x3, stride=4) => ReLU => ResponseNorm(k=2,n=5,a=1e-4,b=0.75) => MaxPool(kernel_size=3, stride=2)
# Conv2(kernels=256, filter=5x5x48, stride=4) => ReLU => ResponseNorm => MaxPool(kernel_size=3, stride=2)
# Conv3(kernels=384, filter=3x3x256, stride=4) => ReLU 
# Conv4(kernels=384, filter=3x3x192, stride=4) => ReLU
# Conv5(kernels=256, filter=3x3x192, stride=4) => ReLU => MaxPool(kernel_size=3, stride=2)

# FC1(4096 neuron) => Dropout(0.5) => ReLU 
# FC2(4096 neuron) => Dropout(0.5) => ReLU
# FC3(1000 neuron) => ReLU => Softmax

# 11/256 = 0.042
# 0.042 * 32 = 1.3 ~ 3

# 11/2 ~ 4
# 3/2 ~ 1

# 256 / 32 = 8
# all kernels reduced by /3, /2, or /4

# endregion

@ModelRegistry.register('alexnet')
class AlexNetOriginal(nn.Module):
    def __init__(self):
        super().__init__() # [64, 3, 224, 224]

        # Conv1(kernels=96, filter=11x11x3, stride=4) => ReLU => ResponseNorm(k=2,n=5,a=1e-4,b=0.75) => MaxPool(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2) # ([4, 96, 55, 55])
        self.relu1 = nn.ReLU()
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2) # ([4, 96, 27, 27])

        # Conv2(kernels=128, filter=5x5x32, stride=4) => ReLU => ResponseNorm => MaxPool(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, groups=2, padding=2) # ([4, 256, 27, 27])
        self.relu2 = nn.ReLU()
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2) # ([4, 256, 13, 13])

        # Conv3(kernels=192, filter=3x3x128, stride=1) => ReLU 
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1) # ([64, 384, 13, 13])
        self.relu3 = nn.ReLU()

        # Conv4(kernels=192, filter=3x3x96, stride=1) => ReLU
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, groups=2, padding=1) # ([64, 384, 13, 13])
        self.relu4 = nn.ReLU()

        # Conv5(kernels=256, filter=3x3x96, stride=4) => ReLU => MaxPool(kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, stride=1, groups=2, padding=1) # ([64, 256, 13, 13])
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2) # ([64, 256, 6, 6])

        # A linear layer is equivalent to convolution with kernel_size=1
        # FC1(128 neuron) => ReLU
        # self.fc6 = nn.Conv1d(6272, 128, kernel_size=1, groups=2) # ([4, 8192, 1])
        self.fc6 = nn.Linear(256*6*6, 4096)
        self.dropout6 = nn.Dropout()
        self.relu6 = nn.ReLU()

        # FC2(128 neuron) => ReLU
        # self.fc7 = nn.Conv1d(128, 128, kernel_size=1, groups=2) # ([4, 128])
        self.fc7 = nn.Linear(4096, 4096)
        self.dropout7 = nn.Dropout()
        self.relu7 = nn.ReLU()

        # FC3(10 neuron)
        self.fc = nn.Linear(4096, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)

        x = torch.flatten(x, 1)

        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        x = self.fc7(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc(x)

        #TODO:
        # data augmentation pca
        # feature activation vectors and visualization

        # PAPER_SUGGESTION:
        # Compute similarity using an auto-encoder rather than Euclidean distance
        # Unsupervised pre-training
            
        return x

@ModelRegistry.register('alexnet_cifar10')
class AlexNet(nn.Module):
    def __init__(self):
        super().__init__() # ([4, 3, 32, 32])

        # Conv1(kernels=64, filter=3x3x3, stride=1) => ReLU => ResponseNorm(k=2,n=5,a=1e-4,b=0.75) => MaxPool(kernel_size=2, stride=1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=2, stride=1, padding=1) # ([4, 64, 32, 32])
        self.relu1 = nn.ReLU()
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2) # ([4, 64, 15, 15])

        # Conv2(kernels=128, filter=5x5x32, stride=4) => ReLU => ResponseNorm => MaxPool(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, groups=2, padding=1) # ([4, 128, 8, 8])
        self.relu2 = nn.ReLU()
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2) # ([4, 128, 3, 3])

        # Conv3(kernels=192, filter=3x3x128, stride=1) => ReLU 
        self.conv3 = nn.Conv2d(128, 192, kernel_size=2, stride=1, padding=1) # ([4, 192, 1, 1])
        self.relu3 = nn.ReLU()

        # Conv4(kernels=192, filter=3x3x96, stride=1) => ReLU
        self.conv4 = nn.Conv2d(192, 192, kernel_size=3, stride=1, groups=2, padding=1)
        self.relu4 = nn.ReLU()

        # Conv5(kernels=256, filter=3x3x96, stride=4) => ReLU => MaxPool(kernel_size=3, stride=2)
        self.conv5 = nn.Conv2d(192, 128, kernel_size=3, stride=1, groups=2, padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 4x4 output here

        # A linear layer is equivalent to convolution with kernel_size=1
        # FC1(128 neuron) => ReLU
        self.fc6 = nn.Linear(2048, 256)
        self.dropout6 = nn.Dropout()
        self.relu6 = nn.ReLU()

        # FC2(128 neuron) => ReLU
        # self.fc7 = nn.Conv1d(128, 128, kernel_size=1, groups=2) # ([4, 128])
        self.fc7 = nn.Linear(256, 256)
        self.dropout7 = nn.Dropout()
        self.relu7 = nn.ReLU()

        # FC3(10 neuron) => ReLU => Softmax
        self.fc = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.lrn2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.conv5(x)
        x = self.relu5(x)

        x = self.maxpool5(x)

        x = torch.flatten(x, 1)

        x = self.fc6(x)
        x = self.dropout6(x)
        x = self.relu6(x)
        x = self.fc7(x)
        x = self.dropout7(x)
        x = self.relu7(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
