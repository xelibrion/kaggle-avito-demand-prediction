import os
import hashlib

import numpy as np
import luigi
import h5py
from sklearn.externals import joblib

import torch

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

from .nnet import Tuner
from .compose_nnet import create_model, create_data_pipeline, gpu_accelerated


class PrecomputedFeature(luigi.ExternalTask):
    feature_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}.pkl')


class ComposeDataset(luigi.Task):
    fold_idx = luigi.IntParameter()
    features = luigi.Parameter()
    target_column = luigi.Parameter()

    def requires(self):
        features = self.features.split(',')
        return {'features': [PrecomputedFeature(x) for x in features]}

    def output(self):
        hash_object = hashlib.md5(self.features.encode('utf-8'))
        params_hash = hash_object.hexdigest()[:6]

        return luigi.LocalTarget(f'data_{self.fold_idx}_{params_hash}.h5')

    def _load_features(self):
        feature_sets = []
        for f_task in self.input()['features']:
            feature_sets.append(joblib.load(f_task.path))

        return np.hstack(feature_sets)

    def _get_train_test(self):
        with h5py.File(self.input()['folds'].path, 'r') as in_file:
            train_idx = in_file['train'][f'fold_{self.fold_idx}'].value
            test_idx = in_file['test'][f'fold_{self.fold_idx}'].value
            return train_idx, test_idx

    def _get_dtype(self, features):
        if np.issubdtype(features.dtype, np.str):
            return features.astype('S'), h5py.special_dtype(vlen=str)

        return features, features.dtype

    def _write_dataset(self, label, features, target, out_file):
        grp = out_file.create_group(label)
        features, dtype = self._get_dtype(features)
        grp.create_dataset(
            'features',
            features.shape,
            data=features,
            dtype=dtype,
            compression='gzip',
            compression_opts=1,
        )
        grp.create_dataset(
            'target',
            target.shape,
            data=target,
            compression='gzip',
            compression_opts=1,
        )

    def run(self):
        features = self._load_features()
        train_idx, test_idx = self._get_train_test()

        df = pd.read_csv(self.input()['raw_data'].path)
        target_values = df[self.target_column].values

        train_set = features[train_idx]
        train_set_target = target_values[train_idx]
        test_set = features[test_idx]
        test_set_target = target_values[test_idx]

        try:
            with h5py.File(self.output().path, 'w') as out_file:
                self._write_dataset('train', train_set, train_set_target, out_file)
                self._write_dataset('test', test_set, test_set_target, out_file)
        except:  # noqa pylint disable=bare-except
            os.remove(self.output().path)
            raise


class TrainNNetOnFold(luigi.Task):
    fold_idx = luigi.IntParameter()
    target = luigi.Parameter()
    lr = luigi.FloatParameter(default=0.001)
    batch_size = luigi.IntParameter(default=32)
    bootstrap_batches = luigi.IntParameter(default=100)

    resources = {'train_concurrency': 1}

    def requires(self):
        return ComposeDataset(
            fold_idx=self.fold_idx,
            features='image',
            target_column=self.target,
        )

    def output(self):
        pass

    def _get_input_data(self, label):
        with h5py.File(self.input().path, 'r') as in_file:
            features = in_file[label]['features'].value
            target = in_file[label]['target'].value
            return features, target

    def run(self):
        cudnn.benchmark = True

        train_images, train_targets = self._get_input_data('train')
        test_images, test_targets = self._get_input_data('test')

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
