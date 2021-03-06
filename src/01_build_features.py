#!/usr/bin/env python

import argparse
import logging

import luigi
from luigi.interface import setup_interface_logging
from pipeline.feature_eng import (CorrectImagePath, ApplyLogTransform, MarkNullInstances,
                                  FillNaTransform, CreateFolds, TrainSet, OneHotEncode,
                                  CharVocabulary, StdScaled, ExtractFeature, LabelEncode,
                                  FastTextVectors, WordVectors)

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

logging.getLogger("luigi.scheduler").setLevel(logging.WARNING)


class GenerateFeatures(luigi.WrapperTask):
    id_column = luigi.Parameter(default='item_id')
    dataset = luigi.TaskParameter(default=TrainSet())

    text_features = [
        'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2',
        'param_3', 'title', 'description', 'user_type'
    ]

    def requires(self):
        yield self.clone(CorrectImagePath, feature_name='image')
        yield self.clone(ApplyLogTransform, feature_name='deal_probability')
        yield self.clone(MarkNullInstances, feature_name='price')
        yield self.clone(FillNaTransform, feature_name='price')
        yield self.clone(StdScaled, feature_name='price_fillna')
        yield self.clone(MarkNullInstances, feature_name='image_top_1')
        yield self.clone(FillNaTransform, feature_name='image_top_1')
        yield self.clone(StdScaled, feature_name='image_top_1_fillna')
        yield self.clone(ExtractFeature, feature_name='city')
        yield self.clone(CreateFolds)
        yield self.clone(OneHotEncode, feature_name='user_type')
        yield self.clone(OneHotEncode, feature_name='parent_category_name')
        yield self.clone(OneHotEncode, feature_name='category_name')
        yield self.clone(OneHotEncode, feature_name='region')
        yield self.clone(LabelEncode, feature_name='city')
        yield self.clone(LabelEncode, feature_name='param_1')
        yield self.clone(LabelEncode, feature_name='param_2')
        yield self.clone(LabelEncode, feature_name='param_3')
        yield self.clone(CharVocabulary, feature_name='description')
        yield self.clone(
            WordVectors,
            feature_name='description',
            train_features=','.join(self.text_features))


if __name__ == '__main__':
    setup_interface_logging.has_run = True

    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    tasks = [GenerateFeatures()]
    luigi.build(tasks, workers=12, local_scheduler=True)
