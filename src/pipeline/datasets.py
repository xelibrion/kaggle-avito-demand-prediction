import os
import hashlib

import numpy as np
import pandas as pd
import luigi
from sklearn.externals import joblib
import h5py
import xgboost as xgb

from .input_data import TrainSet, WithImagesSet
from .feature_eng.categorical import EncodeCategoryTrain
from .feature_eng.paths import CorrectImagePath
from .folds import CreateFolds
from .feature_desc import CATEGORICAL


class FeatureHandling(luigi.Task):
    feature_name = luigi.Parameter()

    feature_processing_map = {'image': CorrectImagePath}

    def requires(self):
        if self.feature_name in CATEGORICAL:
            return EncodeCategoryTrain(self.feature_name)

        task_cls = self.feature_processing_map[self.feature_name]
        return task_cls(feature_name=self.feature_name, dataset=WithImagesSet(TrainSet()))
        # return RawFeatureValues(feature_name=self.feature_name, dataset=TrainSet)

    def output(self):
        return self.input()


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
