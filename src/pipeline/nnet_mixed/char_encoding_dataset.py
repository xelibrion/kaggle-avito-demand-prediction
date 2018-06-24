import numpy as np
import torch
from torch.utils.data import dataset


class CharEncodingDataset(dataset.Dataset):
    def __init__(self, text_array, targets, vocabulary, pad_to_length=3200):
        assert len(text_array) == len(targets)

        self.text_array = text_array
        self.targets = targets
        self.vocabulary = vocabulary
        self.pad_to_length = pad_to_length

    def __len__(self):
        return len(self.text_array)

    def __getitem__(self, idx):
        sentence = self.text_array[idx]
        assert len(sentence) == 1

        if not isinstance(sentence[0], float):
            encoded_sentence = [self.vocabulary[x] for x in sentence[0]]
            num_pad = self.pad_to_length - len(encoded_sentence)
            encoded_sentence += [self.vocabulary['<pad>'] for _ in range(num_pad)]
        else:
            encoded_sentence = [
                self.vocabulary['<pad>'] for _ in range(self.pad_to_length)
            ]

        x_tensor = torch.LongTensor(encoded_sentence)
        y_tensor = torch.FloatTensor([self.targets[idx]])

        return x_tensor, y_tensor
