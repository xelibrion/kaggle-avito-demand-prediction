import itertools
import pandas as pd
import luigi
from sklearn.model_selection import KFold
from sklearn.externals import joblib


class CreateFolds(luigi.Task):
    dataset = luigi.TaskParameter()
    num_folds = luigi.IntParameter(default=5)

    def requires(self):
        return self.dataset

    def output(self):
        params = itertools.product(
            range(self.num_folds),
            ['train', 'val'],
        )
        return [luigi.LocalTarget(f'_folds/fold_{fold_id}_{label}.pkl') for fold_id, label in params]

    def run(self):
        for out in self.output():
            out.makedirs()

        df = pd.read_csv(self.input().path)
        folds = KFold(n_splits=self.num_folds, shuffle=True, random_state=0)

        for fold_id, (train_idx, test_idx) in enumerate(folds.split(df)):
            train_path = self.output()[fold_id * 2].path
            val_path = self.output()[fold_id * 2 + 1].path
            joblib.dump(train_idx, train_path, compress=1)
            joblib.dump(test_idx, val_path, compress=1)
