import torch
from torch.utils.data import dataset


class TransformingDataset(dataset.Dataset):
    def __init__(self, features, targets, transforms):
        for _, v in features.items():
            assert len(v) == len(targets)

        self.features = features
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.targets)

    def _apply(self, key, values):
        if key in self.transforms:
            fn = self.transforms[key]
            return fn(values)

        return values

    def __getitem__(self, idx):
        feature_values = [self._apply(k, v[idx]) for k, v in self.features.items()]

        x_tensor = torch.cat([torch.FloatTensor(x) for x in feature_values])
        y_tensor = torch.FloatTensor([self.targets[idx]])

        return x_tensor, y_tensor
