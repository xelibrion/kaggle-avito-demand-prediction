import pandas as pd
import luigi
from sklearn.externals import joblib


class RawFeatureValues(luigi.Task):
    dataset = luigi.Parameter()
    feature_name = luigi.Parameter()

    def requires(self):
        return self.dataset

    def output(self):
        return luigi.LocalTarget(f'.cache/raw_{self.feature_name}.pkl')

    def run(self):
        self.output().makedirs()

        df = pd.read_csv(self.input().path)
        joblib.dump(df[self.feature_name].values, self.output().path)
