from functools import reduce
from multiprocessing import Pool

import numpy as np
import luigi
from luigi.util import requires
from sklearn.externals import joblib

from ..core import ExtractFeature


def build_vocabulary(s_feature):
    vocabulary = {}

    def look_at_string(text):
        if not isinstance(text, str):
            return

        for ch in text:
            if ch not in vocabulary:
                vocabulary[ch] = len(vocabulary)

    s_feature.apply(look_at_string)

    return vocabulary


def merge_vocabularies(one, other):
    for k, _ in other.items():
        if k not in one:
            one[k] = len(one)
    return one


# pylint: disable=no-member
@requires(ExtractFeature)
class CharVocabulary(luigi.Task):
    resources = {'concurrency': 1}

    num_chunks = luigi.IntParameter(default=12)
    pool_size = luigi.IntParameter(default=12)

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}_char_vocabulary.pkl')

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input().path)

        max_text_length = int(df[self.feature_name].str.len().max())
        print(f"Projected matrix dimensions: {df.shape[0]} x {max_text_length}")

        idx_ranges = np.array_split(df.index, self.num_chunks)
        subsets = list(map(lambda x: df.loc[x, self.feature_name], idx_ranges))
        assert len(
            subsets
        ) == self.num_chunks, f"Expected {len(subsets)} to equal {self.num_chunks}"

        p = Pool(self.pool_size)
        vocabulary_candidates = p.map(build_vocabulary, subsets)

        vocabulary = reduce(merge_vocabularies, vocabulary_candidates, {})
        vocabulary['<pad>'] = len(vocabulary)
        print(f"Vocabulary length: {len(vocabulary)}")
        joblib.dump(vocabulary, self.output().path)


# pylint: enable=no-member
