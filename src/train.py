#!/usr/bin/env python
import logging

import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision import transforms, models
from sklearn.model_selection import train_test_split

from pipeline.nnet import StreamingDataset, Tuner


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


def gpu_accelerated(model, criterion):
    if not torch.cuda.is_available():
        logging.getLogger().warning('CUDA is not available')
        return model, criterion

    return model.cuda(), criterion.cuda()


LEARNING_RATE = 0.001


def get_images():
    with open('../images_sample.txt') as img_file:
        for line in img_file.readlines():
            yield '../input/{}'.format(line.strip('\n'))


def main():
    logging.basicConfig(level=logging.INFO)
    cudnn.benchmark = True

    paths = list(get_images())
    targets = np.ones_like(paths, dtype='int16')
    train_paths, val_paths, train_targets, val_targets = train_test_split(
        paths,
        targets,
        random_state=0,
    )

    model, partial_params, full_params = create_model()
    criterion = torch.nn.BCEWithLogitsLoss(size_average=False)
    model, criterion = gpu_accelerated(model, criterion)

    bootstrap_optimizer = torch.optim.Adam(partial_params, LEARNING_RATE)
    optimizer = torch.optim.Adam(full_params, LEARNING_RATE)

    train_loader, val_loader = create_data_pipeline(
        (train_paths, train_targets),
        (val_paths, val_targets),
        32,
    )

    tuner = Tuner(
        model,
        criterion,
        bootstrap_optimizer,
        optimizer,
        bootstrap_epochs=5,
        tag='fold_{}'.format(0),
    )

    tuner.run(train_loader, val_loader)


if __name__ == '__main__':
    main()
