import os
import hashlib

import numpy as np
from sklearn.externals import joblib
import luigi
import h5py


class PrecomputedFeature(luigi.ExternalTask):
    feature_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}.pkl')


class PrecomputedFold(luigi.ExternalTask):
    fold_idx = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(f'_features/fold_{self.fold_idx}.h5')


class ComposeDataset(luigi.Task):
    fold_idx = luigi.IntParameter()
    features = luigi.Parameter(default='image_path')
    target = luigi.Parameter()

    def requires(self):
        features = self.features.split(',')
        return {
            'features': [PrecomputedFeature(x) for x in features],
            'fold': PrecomputedFold(self.fold_idx),
            'target': PrecomputedFeature(self.target)
        }

    def output(self):
        hash_object = hashlib.md5(self.features.encode('utf-8'))
        params_hash = hash_object.hexdigest()[:6]

        return luigi.LocalTarget(f'data_{self.fold_idx}_{params_hash}.h5')

    def _load_features(self):
        feature_sets = []
        for f_task in self.input()['features']:
            feature_df = joblib.load(f_task.path)
            feature_sets.append(feature_df.iloc[:, 1])

        return np.hstack(feature_sets)

    def _get_train_test(self):
        with h5py.File(self.input()['fold'].path, 'r') as in_file:
            train_idx = in_file['train'].value
            test_idx = in_file['test'].value
            return train_idx, test_idx

    def _get_dtype(self, features):
        if np.issubdtype(features.dtype, np.str):
            return features.astype('S'), h5py.special_dtype(vlen=str)

        return features, features.dtype

    def _write_dataset(self, label, features, target, out_file):
        grp = out_file.create_group(label)
        features, dtype = self._get_dtype(features)

        grp.create_dataset(
            'features',
            features.shape,
            data=features,
            dtype=h5py.special_dtype(vlen=str),
            compression='gzip',
            compression_opts=1,
        )
        print(target.shape)
        print(target[:10])

        grp.create_dataset(
            'target',
            target.shape,
            data=target,
            compression='gzip',
            compression_opts=1,
        )

    def run(self):
        features = self._load_features()

        target = joblib.load(self.input()['target'].path)
        target_values = target.iloc[:, 1].values.astype(float)

        train_idx, test_idx = self._get_train_test()

        train_set = features[train_idx]
        train_set_target = target_values[train_idx]
        test_set = features[test_idx]
        test_set_target = target_values[test_idx]

        try:
            with h5py.File(self.output().path, 'w') as out_file:
                self._write_dataset('train', train_set, train_set_target, out_file)
                self._write_dataset('test', test_set, test_set_target, out_file)
        except:  # noqa pylint disable=bare-except
            os.remove(self.output().path)
            raise
