import torch
from torch.utils.data import dataset


class CharEncodingDataset(dataset.Dataset):
    def __init__(self, features, targets, vocabulary, pad_to_length=3200):
        assert len(features) == len(targets)

        self.features = features
        self.targets = targets
        self.vocabulary = vocabulary
        self.pad_to_length = pad_to_length

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        numeric, sentence = self.features[idx][:-1], self.features[idx][-1]

        if not isinstance(sentence, float):
            encoded_sentence = [self.vocabulary[x] for x in sentence[0]]
            num_pad = self.pad_to_length - len(encoded_sentence)
            encoded_sentence += [self.vocabulary['<pad>'] for _ in range(num_pad)]
        else:
            encoded_sentence = [
                self.vocabulary['<pad>'] for _ in range(self.pad_to_length)
            ]

        x_tensor = torch.cat((torch.LongTensor(numeric.astype(int)),
                              torch.LongTensor(encoded_sentence)))
        y_tensor = torch.FloatTensor([self.targets[idx]])

        return x_tensor, y_tensor
