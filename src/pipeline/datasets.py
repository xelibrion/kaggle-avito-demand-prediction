import numpy as np
import pandas as pd
import luigi
from sklearn.externals import joblib
import h5py

from .tasks import EncodeCategoryTrain, TrainSet
from .folds import CreateTrainFolds


class FeatureHandling(luigi.Task):
    feature_name = luigi.Parameter()

    def requires(self):
        return EncodeCategoryTrain(self.feature_name)

    def output(self):
        return self.input()


class ComposeDataset(luigi.Task):
    fold_idx = luigi.IntParameter()
    features = luigi.Parameter()
    target_column = luigi.Parameter()

    def requires(self):
        feature_steps = [
            FeatureHandling(feature_name=x) for x in self.features.split(',')
        ]
        return {
            'raw_data': TrainSet(),
            'folds': CreateTrainFolds(),
            'features': feature_steps,
        }

    def output(self):
        return luigi.LocalTarget(f'data_{self.fold_idx}.h5')

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

    def _write_dataset(self, label, features, target, out_file):
        grp = out_file.create_group(label)
        grp.create_dataset(
            'features',
            features.shape,
            data=features,
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

        out_file = h5py.File(self.output().path, 'w')
        self._write_dataset('train', train_set, train_set_target, out_file)
        self._write_dataset('test', test_set, test_set_target, out_file)

        out_file.close()


class TrainOnFold(luigi.Task):
    fold_idx = luigi.IntParameter()
    features = luigi.Parameter()
    target = luigi.Parameter()

    def requires(self):
        return ComposeDataset(
            fold_idx=self.fold_idx,
            features=self.features,
            target_column=self.target,
        )

    def output(self):
        return luigi.LocalTarget('model.xgb')

    def run(self):
        print('xgb.train')
