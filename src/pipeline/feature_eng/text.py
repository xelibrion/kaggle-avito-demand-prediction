import json
import itertools
from functools import reduce

import numpy as np
import pandas as pd
import luigi
from luigi.util import requires
from sklearn.externals import joblib
from multiprocessing import Pool

from .core import ExtractFeature


def json_dump(obj, path):
    with open(path, 'w', encoding='utf-8') as out_file:
        json.dump(obj, out_file)


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
    for k, v in other.items():
        if k not in one:
            one[k] = len(one)
    return one


def encode_series(input_tuple):
    s_feature, vocabulary = input_tuple

    def encode_string(text):
        if not isinstance(text, str):
            return []

        return [vocabulary[ch] for ch in text]

    return s_feature.apply(encode_string)


@requires(ExtractFeature)
class CharEncode(luigi.Task):

    resources = {'concurrency': 1}
    pool_size = luigi.IntParameter(default=12)

    def output(self):
        return {
            'result': luigi.LocalTarget(f'_features/{self.feature_name}_char_enc.pkl'),
            'vocabulary': luigi.LocalTarget(f'_reference/{self.feature_name}_char_enc_vocabulary.json'),
        }

    def run(self):
        for _, out in self.output().items():
            out.makedirs()

        df = joblib.load(self.input().path)

        max_text_length = int(df[self.feature_name].str.len().max())
        print(f"Projected matrix dimensions: {df.shape[0]} x {max_text_length}")

        idx_ranges = np.array_split(df.index, self.pool_size)
        subsets = list(map(lambda x: df.loc[x, self.feature_name], idx_ranges))
        assert len(subsets) == self.pool_size, f"Expected {len(subsets)} to equal {self.pool_size}"

        p = Pool(self.pool_size)
        vocabulary_candidates = p.map(build_vocabulary, subsets)

        vocabulary = reduce(merge_vocabularies, vocabulary_candidates, {})
        print(f"Vocabulary length: {len(vocabulary)}")

        input_pairs = list(itertools.product(subsets, [vocabulary]))
        assert len(input_pairs) == self.pool_size, f"Expected {len(input_pairs)} to equal {self.pool_size}"

        encoded_series = p.map(encode_series, input_pairs)
        print(encoded_series[0].head())

        enc_df = encoded_series[0].apply(lambda x: pd.Series(x, dtype='int16'))
        # df = df[[self.id_column]].join(enc_df)

        joblib.dump(enc_df, self.output()['result'].path, compress=1)
        json_dump(vocabulary, self.output()['vocabulary'].path)
