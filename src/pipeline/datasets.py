import numpy as np
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

    def requires(self):
        feature_steps = [
            FeatureHandling(feature_name=x) for x in self.features.split(',')
        ]
        return {
            'train': TrainSet(),
            'folds': CreateTrainFolds(),
            'features': feature_steps,
        }

    def output(self):
        return luigi.LocalTarget(f'data_{self.fold_idx}.h5')

    def run(self):
        feature_sets = []
        for f_task in self.input()['features']:
            feature_sets.append(joblib.load(f_task.path))

        with h5py.File(self.input()['folds'].path, 'r') as in_file:
            train_idx = in_file['train'][f'fold_{self.fold_idx}'].value
            test_idx = in_file['test'][f'fold_{self.fold_idx}'].value

        full_set = np.hstack(feature_sets)
        train_set = full_set[train_idx]
        test_set = full_set[test_idx]

        out_file = h5py.File(self.output().path, 'w')
        train_grp = out_file.create_group('train')
        train_grp.create_dataset(
            'features',
            train_set.shape,
            data=train_set,
            compression='gzip',
        )

        test_grp = out_file.create_group('test')
        test_grp.create_dataset(
            'features',
            test_set.shape,
            data=test_set,
            compression='gzip',
        )

        out_file.close()


class TrainOnFold(luigi.Task):
    fold_idx = luigi.IntParameter()
    features = luigi.Parameter()
    target = luigi.Parameter()

    def requires(self):
        return {
            'features': ComposeDataset(
                fold_idx=self.fold_idx,
                features=self.features,
            )
        }

    def output(self):
        pass

    def run(self):
        print('xgb.train')
