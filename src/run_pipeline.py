#!/usr/bin/env python

import argparse
import logging

import luigi
from luigi.interface import setup_interface_logging
from pipeline import TrainOnFold

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logging.getLogger("luigi.scheduler").setLevel(logging.WARNING)

if __name__ == '__main__':
    setup_interface_logging.has_run = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument('--features', default='region,city')
    parser.add_argument('--target', default='deal_probability')

    args = parser.parse_args()

    tasks = [TrainOnFold(fold_idx=args.fold, features=args.features, target=args.target)]

    luigi.build(tasks, workers=12, local_scheduler=True)
