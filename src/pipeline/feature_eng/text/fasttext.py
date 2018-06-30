import os
import hashlib
from functools import reduce

import luigi
from luigi.util import requires, inherits
from sklearn.externals import joblib
from plumbum import local, FG

from ..core import ExtractFeature, CoreParams


def concat_rowwise(x, y):
    return x + '|' + y


@inherits(CoreParams)
class FastTextInput(luigi.Task):
    features = luigi.Parameter()

    @property
    def features_as_array(self):
        return self.features.split(',')

    def requires(self):
        return [
            self.clone(ExtractFeature, feature_name=x) for x in self.features_as_array
        ]

    def output(self):
        hash_content = f'{self.features}|{self.id_column}|{self.dataset}'
        hash_object = hashlib.md5(hash_content.encode('utf-8'))
        digest = hash_object.hexdigest()[:6]
        return luigi.LocalTarget(f'_features/fasttext_input_{digest}.txt')

    def run(self):
        self.output().makedirs()

        dfs = [joblib.load(x.path) for x in self.input()]
        s_features = [df[x].fillna('') for df, x in zip(dfs, self.features_as_array)]
        joined_rows = reduce(concat_rowwise, s_features)
        with open(self.output().path, 'w+') as out_file:
            out_file.write('\n\n#####\n'.join(joined_rows))


@requires(FastTextInput)
class FastTextVectors(luigi.Task):
    fasttext_path = luigi.Parameter(
        significant=False,
        default='~/projects/fasttext/fasttext',
    )

    def output(self):
        hash_content = f'{self.features}|{self.id_column}|{self.dataset}'
        hash_object = hashlib.md5(hash_content.encode('utf-8'))
        digest = hash_object.hexdigest()[:6]
        return luigi.LocalTarget(f'_features/fasttext_model_{digest}.vec')

    def run(self):
        self.output().makedirs()

        fasttext = local[self.fasttext_path]
        out_path, _ = os.path.splitext(self.output().path)

        fasttext['skipgram', '-input',
                 self.input().path, '-output', out_path, '-minn', '1'] & FG
