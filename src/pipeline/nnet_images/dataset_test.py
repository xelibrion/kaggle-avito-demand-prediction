import unittest
import torch
from .dataset import StreamingDataset


class StreamingDatasetTests(unittest.TestCase):
    def test_when_called_should_return_correct_tensor_types(self):
        ds = StreamingDataset(['./sample_img.png'], [0])
        x_tensor, y_tensor = ds[0]
        self.assertIsInstance(x_tensor, torch.ByteTensor)
        self.assertIsInstance(y_tensor, torch.DoubleTensor)
