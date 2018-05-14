#!/usr/bin/env python

import argparse
import logging

import luigi
from luigi.interface import setup_interface_logging
from pipeline import TrainOnFold

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logging.getLogger("luigi.scheduler").setLevel(logging.WARNING)


class AllInOne(luigi.Task):
    def requires(self):
        yield TrainOnFold()

    def complete(self):
        return all([x.complete() for x in self.requires()])


if __name__ == '__main__':
    setup_interface_logging.has_run = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='AllInOne')
    parser.add_argument('--fold', required=True, type=int)
    parser.add_argument('--features', default='region,city')
    parser.add_argument('--target', default='deal_probability')

    args, unknown = parser.parse_known_args()

    luigi.build(
        [TrainOnFold(
            fold_idx=args.fold,
            features=args.features,
            target=args.target,
        )],
        workers=12,
        local_scheduler=True)
