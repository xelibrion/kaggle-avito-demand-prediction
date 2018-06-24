import logging
import torch

from .char_encoding_dataset import CharEncodingDataset
from .model import MixedNet


def gpu_accelerated(model, criterion):
    if not torch.cuda.is_available():
        logging.getLogger().warning('CUDA is not available')
        return model, criterion

    return model.cuda(), criterion.cuda()


def create_data_pipeline(train_set, val_set, vocabulary, batch_size, workers=10):
    train_features, train_targets = train_set
    print()
    print(train_features[:5], end='\n\n')

    val_features, val_targets = val_set

    train_loader = torch.utils.data.DataLoader(
        CharEncodingDataset(train_features, train_targets, vocabulary),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        CharEncodingDataset(val_features, val_targets, vocabulary),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def create_model(description_voc_size):
    model = MixedNet(description_voc_size=description_voc_size)

    return model, model.parameters()
