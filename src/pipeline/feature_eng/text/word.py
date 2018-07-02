import re
import luigi
from luigi.util import inherits
from plumbum import local, FG
from sklearn.externals import joblib
from nltk.tokenize import RegexpTokenizer

from ..core import SingleFeatureParams, ExtractFeature
from .fasttext import FastTextVectors


@inherits(SingleFeatureParams)
class WordVectors(luigi.Task):
    fasttext_path = luigi.Parameter(
        significant=False,
        default='~/projects/fasttext/fasttext',
    )
    train_features = luigi.Parameter()

    tokenizer = RegexpTokenizer(
        r'\w+|[^\w\s]+', flags=re.DOTALL | re.MULTILINE | re.UNICODE)

    def requires(self):
        return {
            'fasttext_model': self.clone(FastTextVectors, features=self.train_features),
            'feature': self.clone(ExtractFeature),
        }

    def output(self):
        return luigi.LocalTarget('./bla')

    def run(self):
        fasttext = local[self.fasttext_path]

        df = joblib.load(self.input()['feature'].path)

        for sentence in df[self.feature_name].fillna('').values:
            if not sentence:
                continue

            for word in self.tokenizer.tokenize(sentence):
                print(word)
