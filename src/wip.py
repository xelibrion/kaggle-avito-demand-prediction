#!/usr/bin/env python

import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.externals import joblib

import luigi
from luigi.interface import setup_interface_logging
from luigi.util import requires
from h5dataset import h5_dump, h5_load


class PrecomputedFeature(luigi.ExternalTask):
    feature_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}.pkl')


class PrecomputedFold(luigi.ExternalTask):
    fold_id = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(f'_features/fold_{self.fold_id}.h5')


class FilterFeatureToFold(luigi.Task):
    feature_name = luigi.Parameter()
    fold_id = luigi.IntParameter()

    def requires(self):
        return {
            'feature': self.clone(PrecomputedFeature),
            'fold': self.clone(PrecomputedFold),
        }

    def output(self):
        return luigi.LocalTarget(f'_folds/{self.feature_name}_{self.fold_id}.pkl')

    def run(self):
        df = joblib.load(self.input()['feature'].path)


class ComposeDataset(luigi.Task):
    fold_id = luigi.IntParameter()
    features = luigi.Parameter()
    target = luigi.Parameter()

    def requires(self):
        features = self.features.split(',')
        return {
            'features': [self.clone(FilterFeatureToFold, feature_name=x) for x in features],
            'target': PrecomputedFeature(self.target)
        }

    def run(self):
        pass


@requires(ComposeDataset)
class TrainNNetOnFold(luigi.Task):
    features = luigi.Parameter()
    target = luigi.Parameter()
    fold_id = luigi.IntParameter()

    lr = luigi.FloatParameter(default=0.001)
    batch_size = luigi.IntParameter(default=32)
    bootstrap_batches = luigi.IntParameter(default=100)

    resources = {'train_concurrency': 1}

    def run(self):

        train_images, train_targets = self._get_input_data('train')
        test_images, test_targets = self._get_input_data('test')

        print(train_images[:5])
        print(type(train_images))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("luigi.scheduler").setLevel(logging.WARNING)

    setup_interface_logging.has_run = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', default=1)
    parser.add_argument('--features', default='user_type_ohe,region_ohe')
    parser.add_argument('--target', default='deal_probability_log')
    parser.add_argument('--batch-size', type=int, default=64)

    args = parser.parse_args()

    tasks = [
        TrainNNetOnFold(
            fold_id=args.fold - 1,
            features=args.features,
            target=args.target,
            batch_size=args.batch_size,
        ),
    ]

    luigi.build(tasks, workers=12, local_scheduler=True)


if __name__ == '__main__':
    main()