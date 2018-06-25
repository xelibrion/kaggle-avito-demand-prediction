import numpy as np
import pandas as pd
import luigi
from luigi.util import requires
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from .core import ExtractFeature


@requires(ExtractFeature)
class MarkNullInstances(luigi.Task):
    dataset = luigi.TaskParameter()
    feature_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}_isnull.pkl')

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input().path)
        df.iloc[:, 1] = pd.isnull(df.iloc[:, 1])

        joblib.dump(df, self.output().path)


@requires(ExtractFeature)
class FillNaTransform(luigi.Task):
    dataset = luigi.TaskParameter()
    feature_name = luigi.Parameter()

    fill_value = luigi.FloatParameter(default=-999)

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}_fillna.pkl')

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input().path)
        df.iloc[:, 1] = df.iloc[:, 1].fillna(self.fill_value)

        joblib.dump(df, self.output().path)


@requires(ExtractFeature)
class ApplyLogTransform(luigi.Task):
    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}_log.pkl')

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input().path)
        df.iloc[:, 1] = np.log(df.iloc[:, 1] + 0.001)

        joblib.dump(df, self.output().path)


@requires(ExtractFeature)
class StdScaled(luigi.Task):
    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}_stdscaled.pkl')

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input().path)
        scaler = StandardScaler()
        df.iloc[:, 1] = scaler.fit_transform(df.iloc[:, 1].values.reshape(-1, 1))

        joblib.dump(df, self.output().path)
