import pandas as pd
import luigi
from sklearn.externals import joblib


class CommonParams(luigi.Task):
    dataset = luigi.TaskParameter()
    id_column = luigi.Parameter()
    feature_name = luigi.Parameter()


class ExtractFeature(luigi.Task):
    dataset = luigi.TaskParameter()

    id_column = luigi.Parameter()
    feature_name = luigi.Parameter()

    def requires(self):
        return self.dataset

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}.pkl')

    def run(self):
        self.output().makedirs()

        df = pd.read_csv(self.input().path, usecols=[self.id_column, self.feature_name])
        joblib.dump(df, self.output().path, compress=1)
