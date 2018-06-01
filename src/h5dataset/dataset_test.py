import numpy as np
import unittest
import tempfile

from .dataset import dump, load


class DatasetTests(unittest.TestCase):
    def test_should_handle_single_dataset(self):
        with tempfile.NamedTemporaryFile() as target_file:
            out_path = target_file.name

            data = np.random.random_sample(10)
            dump({'train': data}, out_path)

            loaded_data = load(out_path, 'train')
            np.testing.assert_array_equal(data, loaded_data)

    def test_should_handle_multiple_datasets(self):
        with tempfile.NamedTemporaryFile() as target_file:
            out_path = target_file.name

            train = np.random.random_sample(10)
            val = np.random.random_sample(10)
            dump({'train': train, 'val': val}, out_path)

            train_l, val_l = load(out_path, ['train', 'val'])
            np.testing.assert_array_equal(train, train_l)
            np.testing.assert_array_equal(val, val_l)
