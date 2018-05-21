import luigi
import h5py
from .datasets import ComposeDataset


class TrainNNetOnFold(luigi.Task):
    fold_idx = luigi.IntParameter()
    features = luigi.Parameter()
    target = luigi.Parameter()

    resources = {'train_concurrency': 1}

    def requires(self):
        return ComposeDataset(
            fold_idx=self.fold_idx,
            features=self.features,
            target_column=self.target,
        )

    def output(self):
        return luigi.LocalTarget(f'model_{self.fold_idx}.xgb')

    def _get_input_data(self, label):
        with h5py.File(self.input().path, 'r') as in_file:
            features = in_file[label]['features'].value
            target = in_file[label]['target'].value
            return features, target

    def run(self):
        pass
