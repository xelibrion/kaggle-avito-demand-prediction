import os
import hashlib

import numpy as np
import pandas as pd
import luigi
from sklearn.externals import joblib
import h5py
import xgboost as xgb

from .input_data import TrainSet
from .feature_eng.categorical import EncodeCategoryTrain
from .feature_eng.paths import CorrectImagePath
from .folds import CreateTrainFolds
from .feature_desc import CATEGORICAL


class FeatureHandling(luigi.Task):
    feature_name = luigi.Parameter()

    feature_processing_map = {'image': CorrectImagePath}

    def requires(self):
        if self.feature_name in CATEGORICAL:
            return EncodeCategoryTrain(self.feature_name)

        task_cls = self.feature_processing_map[self.feature_name]
        return task_cls(feature_name=self.feature_name, dataset=TrainSet())
        # return RawFeatureValues(feature_name=self.feature_name, dataset=TrainSet)

    def output(self):
        return self.input()


class ComposeDataset(luigi.Task):
    fold_idx = luigi.IntParameter()
    features = luigi.Parameter()
    target_column = luigi.Parameter()

    def requires(self):
        feature_steps = [FeatureHandling(feature_name=x) for x in self.features.split(',')]
        return {
            'raw_data': TrainSet(),
            'folds': CreateTrainFolds(),
            'features': feature_steps,
        }

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


class TrainOnFold(luigi.Task):
    fold_idx = luigi.IntParameter()
    features = luigi.Parameter()
    target = luigi.Parameter()

    resources = {'train_concurrency': 1}

    def requires(self):
        return ComposeDataset(
            fold_idx=self.fold_idx,
            features=self.features,
            target_column=self.target,
        )

    def output(self):
        return luigi.LocalTarget(f'model_{self.fold_idx}.xgb')

    def _get_input_data(self, label):
        with h5py.File(self.input().path, 'r') as in_file:
            features = in_file[label]['features'].value
            target = in_file[label]['target'].value
            return features, target

    def run(self):
        features, target = self._get_input_data('train')
        test_features, test_target = self._get_input_data('test')

        print(f"Features shape: {features.shape}")

        dtrain = xgb.DMatrix(features, label=target)
        dtest = xgb.DMatrix(test_features, label=test_target)

        param = [
            ('max_depth', 6),
            ('objective', 'reg:linear'),
            ('subsample', 0.8),
            ('tree_method', 'exact'),
            # # learn rate
            ('eta', 0.1),
            ('silent', 1),

            #  of multiple eval metrics the last one is used for early stop
            ('eval_metric', 'rmse'),
        ]

        watchlist = [
            (dtrain, 'train'),
            (dtest, 'test'),
        ]

        bst = xgb.train(
            param,
            dtrain,
            100,
            watchlist,
            verbose_eval=2,
            early_stopping_rounds=10,
        )
        # bst.dump_model('./{}.dump'.format(model_name), with_stats=True)
        bst.save_model(self.output().path)
