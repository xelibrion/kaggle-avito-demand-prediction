import logging
import torch
from torch.utils.data import TensorDataset

from .model import MixedNet


def gpu_accelerated(model, criterion):
    if not torch.cuda.is_available():
        logging.getLogger().warning('CUDA is not available')
        return model, criterion

    return model.cuda(), criterion.cuda()


def create_data_pipeline(train_set, val_set, batch_size, workers=6):
    train_features, train_targets = train_set
    train_features, train_targets = torch.Tensor(train_features), torch.Tensor(train_targets)

    val_features, val_targets = val_set
    val_features, val_targets = torch.Tensor(val_features), torch.Tensor(val_targets)

    train_loader = torch.utils.data.DataLoader(
        TensorDataset((train_features, train_targets)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        TensorDataset((val_features, val_targets)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def create_model(description_voc_size):
    model = MixedNet(description_voc_size=description_voc_size)
    return model, model.parameters()
