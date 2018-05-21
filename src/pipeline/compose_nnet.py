import logging
import torch
import torch.nn as nn
from torchvision import transforms, models

from .nnet import StreamingDataset


def gpu_accelerated(model, criterion):
    if not torch.cuda.is_available():
        logging.getLogger().warning('CUDA is not available')
        return model, criterion

    return model.cuda(), criterion.cuda()


def create_data_pipeline(train_set, val_set, batch_size, workers=6):
    image_paths, targets = train_set
    val_image_paths, val_targets = val_set

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    img_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = torch.utils.data.DataLoader(
        StreamingDataset(image_paths, targets, transform=img_transforms),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        StreamingDataset(val_image_paths, val_targets, transform=img_transforms),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def create_model():
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(512, 1)

    full_params = list(model.layer4.parameters()) + list(model.fc.parameters())
    return model, model.fc.parameters(), full_params
