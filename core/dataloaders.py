from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, random_split


class DataloaderRegistry:
    _registry = {}

    @classmethod
    def register(cls, name):
        def decorator(func):
            cls._registry[name.lower()] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name, **kwargs):
        if name.lower() not in cls._registry:
            raise ValueError(f"Unknown dataloader: {name}")
        return cls._registry[name.lower()](**kwargs)


@DataloaderRegistry.register("cifar10")
def cifar10_loaders(split=[0.9,0.1], batch_size=4, transform=None, test_transform=None):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914,0.4822,0.4465),
                        std=(0.2023,0.1994,0.2010)),
        ])
    
    test_transform = transform if not test_transform else test_transform

    all_data = datasets.CIFAR10(
        root='data',
        train=True,
        transform=transform,
        download=True
    )

    train_data, val_data = random_split(all_data, split)

    test_data = datasets.CIFAR10(
        root='data',
        train=False,
        transform=test_transform,
        download=True
    )


    loader = {
        'train': DataLoader(train_data, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_data, batch_size=batch_size, shuffle=False),
        'test': DataLoader(test_data, batch_size=batch_size, shuffle=False)
    }

    return loader

@DataloaderRegistry.register("cifar10_scaled256")
def cifar10_scaled256_loaders(split=[0.9,0.1], batch_size=4):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465),
                    std=(0.2023,0.1994,0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914,0.4822,0.4465),
                    std=(0.2023,0.1994,0.2010)),
    ])

    return cifar10_loaders(transform=transform, test_transform=test_transform)
