import numpy as np
import pandas as pd
import luigi
from sklearn.externals import joblib


class CorrectImagePath(luigi.Task):
    dataset = luigi.TaskParameter()
    feature_name = luigi.Parameter()

    def requires(self):
        return self.dataset

    def output(self):
        return luigi.LocalTarget(f'.cache/img_{self.feature_name}.pkl')

    def _img_id_to_path(self, img_id):
        return f'../input/images/train/{img_id}.jpg'

    def run(self):
        self.output().makedirs()

        df = pd.read_csv(self.input().path)
        ids = df[self.feature_name].values
        paths = np.vectorize(self._img_id_to_path)(ids)
        joblib.dump(paths, self.output().path)
