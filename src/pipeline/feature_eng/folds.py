import pandas as pd
import luigi
from sklearn.model_selection import KFold
from ..h5dataset import h5_dump


class CreateFolds(luigi.Task):
    dataset = luigi.TaskParameter()
    num_folds = luigi.IntParameter(default=5)

    def requires(self):
        return self.dataset

    def output(self):
        return [luigi.LocalTarget(f'_folds/fold_{fold_id}.h5') for fold_id in range(self.num_folds)]

    def run(self):
        for out in self.output():
            out.makedirs()

        df = pd.read_csv(self.input().path)
        folds = KFold(n_splits=self.num_folds, shuffle=True, random_state=0)

        for fold_id, (train_idx, val_idx) in enumerate(folds.split(df)):
            out_path = self.output()[fold_id].path
            h5_dump({
                'train': train_idx,
                'val': val_idx,
            }, out_path)
