import logging
from functools import partial

import numpy as np
import torch

from .transforming_dataset import TransformingDataset
from .model import MixedNet


def gpu_accelerated(model, criterion):
    if not torch.cuda.is_available():
        logging.getLogger().warning('CUDA is not available')
        return model, criterion

    return model.cuda(), criterion.cuda()


def char_encode(vocabulary, sentence, pad_to_length=3250):
    if isinstance(sentence[0], float):
        return [vocabulary['<pad>'] for _ in range(pad_to_length)]

    encoded_sentence = [vocabulary[x] for x in sentence[0]]
    num_pad = pad_to_length - len(encoded_sentence)
    encoded_sentence += [vocabulary['<pad>'] for _ in range(num_pad)]
    return encoded_sentence


def create_data_pipeline(train_set, val_set, vocabulary, batch_size, workers=10):
    train_features, train_targets = train_set
    val_features, val_targets = val_set

    TRANSFORMS = {
        'price_isnull': lambda x: x.astype(np.uint8),
        'image_top_1_isnull': lambda x: x.astype(np.uint8),
        'description': partial(char_encode, vocabulary),
    }

    train_loader = torch.utils.data.DataLoader(
        TransformingDataset(train_features, train_targets, TRANSFORMS),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        TransformingDataset(val_features, val_targets, TRANSFORMS),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader


def create_model(description_voc_size):
    model = MixedNet()

    return model, model.parameters()
