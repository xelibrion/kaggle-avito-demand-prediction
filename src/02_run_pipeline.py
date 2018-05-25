#!/usr/bin/env python

import argparse
import logging

import luigi
from luigi.interface import setup_interface_logging
from pipeline import TrainNNetOnFold

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logging.getLogger("luigi.scheduler").setLevel(logging.WARNING)

PREDICTORS = ['region', 'parent_category_name', 'category_name', 'param_1', 'param_2', 'user_type']


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


if __name__ == '__main__':
    setup_interface_logging.has_run = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--folds', default=[1], action=ParseNumFolds)
    parser.add_argument('--features', default=','.join(PREDICTORS))
    parser.add_argument('--target', default='deal_probability')
    parser.add_argument('--batch-size', type=int, default=64)

    args = parser.parse_args()

    tasks = [
        TrainNNetOnFold(fold_idx=x - 1, target=args.target, batch_size=args.batch_size) for x in args.folds
    ]
    # tasks = [
    #     TrainOnFold(fold_idx=x - 1, features=args.features, target=args.target)
    #     for x in args.folds
    # ]

    luigi.build(tasks, workers=12, local_scheduler=True)
