import torch
import torch.nn as nn
from core.model_registry import ModelRegistry

# region Original Architecture of VGG-16 - Configuration C
# conv(64 filters, 3x3, stride=1, padding=1)
# relu()
# conv(64 filters, 3x3, stride=1, padding=1)
# relu()
# maxpool(2x2, stride=2)

# conv(128 filters, 3x3, stride=1, padding=1)
# relu()
# conv(128 filters, 3x3, stride=1, padding=1)
# relu()
# maxpool(2x2, stride=2)

# conv(256 filters, 3x3, stride=1, padding=1)
# relu()
# conv(256 filters, 3x3, stride=1, padding=1)
# relu()
# conv(256 filters, 1x1, stride=1)
# relu()
# maxpool(2x2, stride=2)

# conv(512 filters, 3x3, stride=1, padding=1)
# relu()
# conv(512 filters, 3x3, stride=1, padding=1)
# relu()
# conv(512 filters, 1x1, stride=1)
# relu()
# maxpool(2x2, stride=2)

# conv(512 filters, 3x3, stride=1, padding=1)
# relu()
# conv(512 filters, 3x3, stride=1, padding=1)
# relu()
# conv(512 filters, 1x1, stride=1)
# relu()
# maxpool(2x2, stride=2)

# fc(4096)
# relu()
# dropout(0.5)
# fc(4096)
# dropout(0.5)
# relu()
# fc(num_classes)
# softmax

# Training Configuration
# MiniBatch Gradient Descent
# batch=256
# lr=0.01 with scheduled decrease
# momentum=0.9, decay=0.0005

# Single-scale Training vs Multi-scale Training
# endregion

@ModelRegistry.register('vgg16')
class VGG16(nn.Module): # Configuration C
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #([64, 64, 224, 224])
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), #([64, 64, 112, 112])

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #([64, 128, 112, 112])
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), #([64, 128, 56, 56])

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), #([64, 256, 56, 56])
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), #([64, 256, 28, 28])

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), #([64, 512, 28, 28])
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), #([64, 512, 14, 14])

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), #([64, 512, 7, 7])
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

@ModelRegistry.register('vgg16_cifar10')
class VGG16_Adapted(nn.Module): # Configuration C
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1), #([64, 64, 32, 32])
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), #([64, 64, 16, 16])

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #([64, 128, 16, 16])
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), #([64, 128, 8, 8])

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), #([64, 256, 8, 8])
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=1, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), #([64, 256, 4, 4])

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), #([64, 512, 4, 4])
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=1, stride=1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2), #([64, 512, 2, 2])

            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(512, 512, kernel_size=1, stride=1),
            # nn.ReLU(),

            # nn.MaxPool2d(kernel_size=2, stride=2), #([64, 512, 7, 7])
        )

        self.classifier = nn.Sequential(
            nn.Linear(512*2*2, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 10),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

