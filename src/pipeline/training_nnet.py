import luigi
from luigi.util import requires
import h5py

import torch

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from .datasets import ComposeDataset
from .nnet import Tuner
from .compose_nnet import create_model, create_data_pipeline, gpu_accelerated


@requires(ComposeDataset)
class TrainNNetOnFold(luigi.Task):
    fold_idx = luigi.IntParameter()
    target = luigi.Parameter()
    lr = luigi.FloatParameter(default=0.001)
    batch_size = luigi.IntParameter(default=32)
    bootstrap_batches = luigi.IntParameter(default=100)

    resources = {'train_concurrency': 1}

    def _get_input_data(self, label):
        with h5py.File(self.input().path, 'r') as in_file:
            features = in_file[label]['features'].value
            target = in_file[label]['target'].value
            return features, target

    def run(self):
        cudnn.benchmark = True

        train_images, train_targets = self._get_input_data('train')
        test_images, test_targets = self._get_input_data('test')

        print(train_images[:5])
        print(type(train_images))

        criterion = torch.nn.BCEWithLogitsLoss()
        model, partial_params, full_params = create_model()
        model, criterion = gpu_accelerated(model, criterion)

        bootstrap_optimizer = torch.optim.Adam(partial_params, self.lr)
        optimizer = torch.optim.Adam(full_params, self.lr)

        train_loader, val_loader = create_data_pipeline(
            (train_images, train_targets),
            (test_images, test_targets),
            self.batch_size,
        )

        tuner = Tuner(
            model,
            criterion,
            bootstrap_optimizer,
            optimizer,
            bootstrap_batches=self.bootstrap_batches,
            tag='fold_{}'.format(0),
        )

        tuner.run(train_loader, val_loader)
