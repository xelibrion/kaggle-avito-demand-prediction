#!/usr/bin/env python

import argparse
import logging

import pandas as pd
import luigi
from luigi.interface import setup_interface_logging

from sklearn.externals import joblib
from pipeline.h5dataset import h5_load, h5_dump


class PrecomputedFeature(luigi.ExternalTask):
    feature_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}.pkl')


class PrecomputedFold(luigi.ExternalTask):
    fold_id = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(f'_folds/fold_{self.fold_id}.h5')


class FilterFeatureToFold(luigi.Task):
    feature_name = luigi.Parameter()
    fold_id = luigi.IntParameter()
    subset = luigi.Parameter()

    id_column = luigi.Parameter()

    def requires(self):
        return {
            'feature': self.clone(PrecomputedFeature),
            'fold': self.clone(PrecomputedFold),
        }

    def output(self):
        return luigi.LocalTarget(f'_feature_folds/{self.feature_name}_{self.fold_id}_{self.subset}.pkl')

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input()['feature'].path)
        subset_idx = h5_load(self.input()['fold'].path, self.subset)

        subset_df = df.iloc[subset_idx].drop(self.id_column, axis=1)
        joblib.dump(subset_df, self.output().path)


class ComposeDataset(luigi.Task):
    features = luigi.Parameter()
    target = luigi.Parameter()

    fold_id = luigi.IntParameter()
    subset = luigi.Parameter()

    id_column = luigi.Parameter()

    def requires(self):
        features = self.features.split(',')
        return {
            'features': [self.clone(FilterFeatureToFold, feature_name=x) for x in features],
            'target': self.clone(FilterFeatureToFold, feature_name=self.target)
        }

    def output(self):
        digest = '000000'
        return luigi.LocalTarget(f'_feature_folds/combined_{self.fold_id}_{self.subset}_{digest}.h5')

    def run(self):
        self.output().makedirs()

        dfs = [joblib.load(x.path) for x in self.input()['features']]
        df = pd.concat(dfs, axis=1, join='inner')

        target_df = joblib.load(self.input()['target'].path)

        h5_dump({
            'features': df.values,
            'target': target_df.values,
        },
                self.output().path)


class TrainNNetOnFold(luigi.Task):
    features = luigi.Parameter()
    target = luigi.Parameter()
    fold_id = luigi.IntParameter()
    id_column = luigi.IntParameter(default='item_id')

    lr = luigi.FloatParameter(default=0.001)
    batch_size = luigi.IntParameter(default=32)
    bootstrap_batches = luigi.IntParameter(default=100)

    resources = {'train_concurrency': 1}

    def requires(self):
        return {
            'train': self.clone(ComposeDataset, subset='train'),
            'val': self.clone(ComposeDataset, subset='val'),
        }

    def run(self):
        features, target = h5_load(self.input()['train'].path, ['features', 'target'])
        print(features.shape)
        print(target.shape)

        val_features, val_target = h5_load(self.input()['val'].path, ['features', 'target'])
        print(val_features.shape)
        print(val_target.shape)


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
