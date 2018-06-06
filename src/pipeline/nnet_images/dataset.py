from pathlib import Path
import torch
from torch.utils.data import dataset
import cv2


class StreamingDataset(dataset.Dataset):
    def __init__(self, image_paths, targets, transform=None):
        assert len(image_paths) == len(targets)

        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        if not Path(path).exists():
            raise ValueError("Path does not exist", path)

        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if self.transform:
            img = self.transform(img)

        x_tensor = torch.FloatTensor(img)
        y_tensor = torch.FloatTensor([self.targets[idx]])

        return x_tensor, y_tensor
