import json

import luigi
from luigi.util import requires
from sklearn.externals import joblib

from .core import ExtractFeature


def json_dump(obj, path):
    with open(path, 'w', encoding='utf-8') as out_file:
        json.dump(obj, out_file)


@requires(ExtractFeature)
class CharEncode(luigi.Task):
    def output(self):
        return {
            'result': luigi.LocalTarget(f'_features/{self.feature_name}_char_enc.pkl'),
            'vocabulary': luigi.LocalTarget(f'_reference/{self.feature_name}_char_enc_vocabulary.json'),
        }

    def encode_string(self, text, vocabulary):
        if not isinstance(text, str):
            return []

        for ch in text:
            if ch not in vocabulary:
                vocabulary[ch] = len(vocabulary)
            yield vocabulary[ch]

    def run(self):
        for _, out in self.output().items():
            out.makedirs()

        df = joblib.load(self.input().path)

        vocabulary = {}
        df[self.feature_name] = df[self.feature_name].apply(lambda x: list(self.encode_string(x, vocabulary)))

        joblib.dump(df, self.output()['result'].path, compress=1)
        json_dump(vocabulary, self.output()['vocabulary'].path)
