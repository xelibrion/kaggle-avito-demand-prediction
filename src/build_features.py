#!/usr/bin/env python

import argparse
import logging

import luigi
from luigi.interface import setup_interface_logging
from feature_store import (ExtractFeature, CorrectImagePath, ApplyLogTransform, MarkNullInstances,
                           FillNaTransform)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logging.getLogger("luigi.scheduler").setLevel(logging.WARNING)


class TrainSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../input/train.csv')


class GenerateFeatures(luigi.WrapperTask):
    def requires(self):
        yield CorrectImagePath(dataset=TrainSet(), feature_name='image')
        yield ApplyLogTransform(dataset=TrainSet(), feature_name='deal_probability')
        yield MarkNullInstances(dataset=TrainSet(), feature_name='price')
        yield FillNaTransform(dataset=TrainSet(), feature_name='price')
        yield ApplyLogTransform(dataset=TrainSet(), feature_name='price')


if __name__ == '__main__':
    setup_interface_logging.has_run = True

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    tasks = [GenerateFeatures()]
    luigi.build(tasks, workers=12, local_scheduler=True)
