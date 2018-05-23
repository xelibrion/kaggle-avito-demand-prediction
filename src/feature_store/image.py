import numpy as np
import luigi
from luigi.util import requires
from sklearn.externals import joblib

from .core import ExtractFeature


@requires(ExtractFeature)
class CorrectImagePath(luigi.Task):
    dataset = luigi.TaskParameter()
    feature_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'_features/image_path.pkl')

    def _img_id_to_path(self, img_id):
        return f'../input/images/train/{img_id}.jpg'

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input().path)
        img_ids = df.iloc[:, 1]
        img_paths = np.vectorize(self._img_id_to_path)(img_ids)
        df.iloc[:, 1] = img_paths
        joblib.dump(df, self.output().path)
