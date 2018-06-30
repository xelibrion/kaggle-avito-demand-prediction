import pandas as pd
import luigi
from luigi.util import inherits
from sklearn.externals import joblib

from .core import ExtractFeature, SingleFeatureParams
from .input_data import TrainSet, TestSet


@inherits(SingleFeatureParams)
class AllCategoryValues(luigi.Task):
    def requires(self):
        return {
            'train': self.clone(ExtractFeature, dataset=TrainSet()),
            'test': self.clone(ExtractFeature, dataset=TestSet()),
        }

    def output(self):
        return luigi.LocalTarget(f'_reference/{self.feature_name}.txt')

    def run(self):
        self.output().makedirs()

        df_train = joblib.load(self.input()['train'].path)
        df_test = joblib.load(self.input()['test'].path)

        train_vals = df_train[self.feature_name].dropna().unique()
        test_vals = df_test[self.feature_name].dropna().unique()
        vals = set(train_vals).union(set(test_vals))

        with open(self.output().path, 'w', encoding='utf-8') as out_file:
            for val in vals:
                out_file.write(val)
                out_file.write('\n')


@inherits(SingleFeatureParams)
class OneHotEncode(luigi.Task):
    def requires(self):
        return {
            'cat_values': self.clone(AllCategoryValues),
            'feature': self.clone(ExtractFeature),
        }

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}_ohe.pkl')

    def _cat_values(self):
        with open(self.input()['cat_values'].path, encoding='utf-8') as cat_file:
            return [x.strip('\n') for x in cat_file.readlines()]

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input()['feature'].path)

        df[self.feature_name] = pd.Categorical(
            df[self.feature_name],
            categories=self._cat_values(),
        )
        df_enc = pd.get_dummies(df[[self.feature_name]])

        df_out = df[[self.id_column]].join(df_enc)
        joblib.dump(df_out, self.output().path)


@inherits(SingleFeatureParams)
class LabelEncode(luigi.Task):
    def requires(self):
        return {
            'cat_values': self.clone(AllCategoryValues),
            'feature': self.clone(ExtractFeature),
        }

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}_le.pkl')

    def _cat_values(self):
        with open(self.input()['cat_values'].path, encoding='utf-8') as cat_file:
            return [x.strip('\n') for x in cat_file.readlines()]

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input()['feature'].path)

        feature_as_cat = pd.Categorical(
            df[self.feature_name],
            categories=self._cat_values(),
        )
        df[self.feature_name] = feature_as_cat.codes

        joblib.dump(df, self.output().path)
