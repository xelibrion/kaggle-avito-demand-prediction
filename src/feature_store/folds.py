import os
import pandas as pd
import luigi
import h5py
from sklearn.model_selection import KFold


class CreateFolds(luigi.Task):
    dataset = luigi.TaskParameter()
    num_folds = luigi.IntParameter(default=5)

    def requires(self):
        return self.dataset

    def output(self):
        return [luigi.LocalTarget(f'_features/fold_{i}.h5') for i in range(self.num_folds)]

    def _write_fold(self, fold_id, train_idx, test_idx):
        print(f"Writing {fold_id} to file")
        out_path = self.output()[fold_id].path

        with h5py.File(out_path, 'w') as out_file:
            out_file.create_dataset(
                'train',
                train_idx.shape,
                data=train_idx,
            )
            out_file.create_dataset(
                'test',
                test_idx.shape,
                data=test_idx,
            )

    def run(self):
        for out in self.output():
            out.makedirs()

        df = pd.read_csv(self.input().path)
        folds = KFold(n_splits=self.num_folds, shuffle=True, random_state=0)

        for idx, (train_idx, test_idx) in enumerate(folds.split(df)):
            try:
                self._write_fold(idx, train_idx, test_idx)
            except Exception as ex:  # noqa pylint disable=bare-except
                print(ex)
                os.remove(self.output().path)
                raise
