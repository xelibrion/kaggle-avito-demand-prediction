#!/usr/bin/env python

import argparse
import logging

import luigi
from luigi.interface import setup_interface_logging

from pipeline.h5dataset import h5_load
from pipeline.core import ComposeDataset


class ParseNumFolds(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(ParseNumFolds, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if '-' in values:
            bounds = values.split('-')
            assert len(bounds) == 2
            namespace.folds = list(range(int(bounds[0]), int(bounds[1]) + 1))
        else:
            namespace.folds = [int(values)]


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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("luigi.scheduler").setLevel(logging.WARNING)

    setup_interface_logging.has_run = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', default=[1], action=ParseNumFolds)
    parser.add_argument('--features', default='region_ohe,user_type_ohe')
    parser.add_argument('--target', default='deal_probability_log')
    parser.add_argument('--batch-size', type=int, default=64)

    args = parser.parse_args()

    tasks = [
        TrainNNetOnFold(
            fold_id=x - 1,
            features=args.features,
            target=args.target,
            batch_size=args.batch_size,
        ) for x in args.folds
    ]

    luigi.build(tasks, workers=12, local_scheduler=True)
