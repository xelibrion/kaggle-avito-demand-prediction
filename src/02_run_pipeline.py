#!/usr/bin/env python

import argparse
import logging

import luigi
from luigi.interface import setup_interface_logging

import torch
import torch.nn
import torch.backends.cudnn as cudnn
import torch.optim

from pipeline.pkl_dataset import load
from pipeline.core import ComposeDataset
from pipeline.nnet_mixed.bootstrap import (create_data_pipeline, create_model, gpu_accelerated)
from pipeline.nnet_mixed import Tuner


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


class CharVocabulary(luigi.ExternalTask):
    feature_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}_char_vocabulary.pkl')


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
            'train': self.clone(ComposeDataset, dataset_part='train'),
            'val': self.clone(ComposeDataset, dataset_part='val'),
            'vocabulary_description': self.clone(CharVocabulary, feature_name='description'),
        }

    def run(self):
        train_features, train_targets = load(self.input()['train'].path, ['features', 'target'])
        print()
        for k, v in train_features.items():
            print(k, v.shape)
        print(train_targets.shape, end='\n\n')

        val_features, val_targets = load(self.input()['val'].path, ['features', 'target'])
        print()
        for k, v in val_features.items():
            print(k, v.shape)
        print(val_targets.shape, end='\n\n')

        vocabulary = load(self.input()['vocabulary_description'].path)

        cudnn.benchmark = True

        criterion = torch.nn.MSELoss()
        model, params = create_model(len(vocabulary))
        model, criterion = gpu_accelerated(model, criterion)

        optimizer = torch.optim.Adam(params, self.lr)

        train_loader, val_loader = create_data_pipeline(
            (train_features, train_targets),
            (val_features, val_targets),
            vocabulary,
            self.batch_size,
        )

        tuner = Tuner(
            model,
            criterion,
            optimizer,
            tag='fold_{}'.format(0),
        )

        tuner.run(train_loader, val_loader)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("luigi.scheduler").setLevel(logging.WARNING)

    setup_interface_logging.has_run = True

    FEATURES = [
        'region_ohe',
        'user_type_ohe',
        'parent_category_name_ohe',
        'category_name_ohe',
        'price_isnull',
        'price_fillna_stdscaled',
        'image_top_1_isnull',
        'image_top_1_fillna_stdscaled',
        'city_le',
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', default=[1], action=ParseNumFolds)
    parser.add_argument('--features', default=','.join(FEATURES))
    parser.add_argument('--target', default='deal_probability_log')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)

    args = parser.parse_args()

    tasks = [
        TrainNNetOnFold(
            fold_id=x - 1,
            features=args.features,
            target=args.target,
            batch_size=args.batch_size,
            lr=args.lr,
        ) for x in args.folds
    ]

    luigi.build(tasks, workers=12, local_scheduler=True)
