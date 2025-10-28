import os
import argparse
import yaml

import torch.nn as nn
import torch.optim as optim
import torch
import torchmetrics as tm

from core.dataloaders import DataloaderRegistry
from core.config import RunConfig
from core.model_registry import ModelRegistry
import papers # for registry
from core.init_device import device

def get_data_loaders(data_config):
    return DataloaderRegistry.get(data_config.name, batch_size=data_config.batch_size)


def train(model,device,dataloaders,config, pretrained=False):

    def init_weights(m):
        if not isinstance(m, nn.Conv2d) and not isinstance(m, nn.Linear):
            return

        if config.model.init=='kaiming_normal':
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif config.model.init_std:
            nn.init.normal_(m.weight, std=config.model.init_std)

        if config.model.init_bias:
            nn.init.constant_(m.bias, config.model.init_bias)

    model.to(device)
    if not pretrained:
        model.apply(init_weights)


    criterion = nn.CrossEntropyLoss()
    print(f"Creating optimizer with {config.optim}")
    initial_lr = config.optim.lr
    optimizer = optim.SGD(model.parameters(), lr=config.optim.lr, momentum=config.optim.momentum, weight_decay=config.optim.weight_decay)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    train_metric_collection = tm.MetricCollection(
        tm.classification.MulticlassAccuracy(num_classes=10, average=None),
        tm.classification.MulticlassPrecision(num_classes=10, average=None),
        tm.classification.MulticlassF1Score(num_classes=10, average=None),
        tm.classification.MulticlassConfusionMatrix(num_classes=10)
    ).to(device)

    train_tracker = tm.MetricTracker(train_metric_collection, maximize=True)

    val_metric_collection = tm.MetricCollection(
        tm.classification.MulticlassAccuracy(num_classes=10, average=None),
        tm.classification.MulticlassPrecision(num_classes=10, average=None),
        tm.classification.MulticlassF1Score(num_classes=10, average=None),
        tm.classification.MulticlassConfusionMatrix(num_classes=10)
    ).to(device)

    val_tracker = tm.MetricTracker(val_metric_collection, maximize=True)

    tracker = {
        'train': train_tracker,
        'val': val_tracker
    }

    losses = []
    EPOCH = config.train.epochs
    for ep in range(EPOCH):
        for phase in ['train', 'val']:
            if phase=='train':
                model.train()
            else:
                model.eval()

            total_loss = 0

            with torch.set_grad_enabled(phase=='train'):
                for i, (data, label) in enumerate(dataloaders[phase]):
                    data, label = data.to(device), label.to(device)
                    optimizer.zero_grad()

                    logits = model(data)
                    loss = criterion(logits, label)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                    total_loss += loss.item()

                    if phase=='train' and i % 1250 == 0 and i!=0:
                        print(f"Avg Loss at {i-1250}-{i}:{total_loss/1250}")
                        losses.append(total_loss)
                        total_loss = 0
                    

                    tracker[phase].increment()
                    tracker[phase].update(torch.softmax(logits, dim=1), label)
                    
                if phase=='val':
                    print(f"EPOCH:{ep}-Validation Set Loss:{total_loss/len(dataloaders[phase])}")
                    lr_scheduler.step(total_loss/len(dataloaders[phase]))
                    if initial_lr != optimizer.param_groups[0]['lr']:
                        print(f"LR Changed {optimizer.param_groups[0]['lr']}")
                        initial_lr = optimizer.param_groups[0]['lr']
    
    return losses, model, tracker


def test_accuracy(model, test_loader):
    probs, labells = [], []
    for xt, yt in test_loader:
        prob = torch.softmax(model(xt), dim=1)
        probs.append(prob)
        labells.append(yt)

    probs = torch.cat(probs, dim=0)
    labells = torch.cat(labells, dim=0)
    acc = tm.functional.accuracy(probs, labells, task='multiclass', num_classes=10)
    print(f"Test Accuracy: {acc:.4f}")
    return acc


def build_model_name(config):
    parts = []
    
    if getattr(config.model, 'name', None) is not None:
        parts.append(config.model.name)
    if getattr(config.train, 'epochs', None) is not None:
        parts.append(f"ep{config.train.epochs}")
    if getattr(config.model, 'init_std', None) is not None:
        parts.append(f"std{config.model.init_std}")
    if getattr(config.model, 'init_bias', None) is not None:
        parts.append(f"bias{config.model.init_bias}")
    if getattr(config.optim, 'lr', None) is not None:
        parts.append(f"lr{config.optim.lr}")
    
    return '_'.join(parts)


def save_model(model, config):
    model_name = build_model_name(config)
    os.makedirs(f"runs/{model_name}", exist_ok=True)
    model_path = f"runs/{model_name}/{model_name}.pth"
    if os.path.isfile(model_path):
        os.remove(model_path)
    torch.save(model.state_dict(), model_path)


def main(config):
    config = RunConfig(**config)
    torch.manual_seed(config.seed)
    dataloaders = get_data_loaders(config.data)
    model = ModelRegistry.get(config.model.name)()
    losses, model, metric_tracker = train(model, device, dataloaders, config)
    save_model(model, config)
    test_accuracy(model, dataloaders['test'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Trainer')
    parser.add_argument('-c', '--config')
    args = parser.parse_args()

    config_path = args.config
    with open(config_path, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    
    main(cfg)