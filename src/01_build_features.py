#!/usr/bin/env python

import argparse
import logging

import luigi
from luigi.interface import setup_interface_logging
from pipeline.feature_eng import (CorrectImagePath, ApplyLogTransform, MarkNullInstances, FillNaTransform, CreateFolds,
                                  TrainSet, OneHotEncode, CharEncode)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logging.getLogger("luigi.scheduler").setLevel(logging.WARNING)


class GenerateFeatures(luigi.WrapperTask):
    id_column = luigi.Parameter(default='item_id')
    dataset = luigi.TaskParameter(default=TrainSet())

    def requires(self):
        yield self.clone(CorrectImagePath, feature_name='image')
        yield self.clone(ApplyLogTransform, feature_name='deal_probability')
        yield self.clone(MarkNullInstances, feature_name='price')
        yield self.clone(FillNaTransform, feature_name='price')
        yield self.clone(ApplyLogTransform, feature_name='price')
        yield self.clone(ApplyLogTransform, feature_name='price')
        yield self.clone(CreateFolds)
        yield self.clone(OneHotEncode, feature_name='user_type')
        yield self.clone(OneHotEncode, feature_name='parent_category_name')
        yield self.clone(OneHotEncode, feature_name='region')
        yield self.clone(CharEncode, feature_name='description')


if __name__ == '__main__':
    setup_interface_logging.has_run = True

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    tasks = [GenerateFeatures()]
    luigi.build(tasks, workers=12, local_scheduler=True)
