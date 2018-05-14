import pandas as pd
import luigi
import h5py
from sklearn.model_selection import KFold

from .tasks import TrainSet


class CreateTrainFolds(luigi.Task):
    num_folds = luigi.IntParameter(default=5)

    def requires(self):
        return TrainSet()

    def output(self):
        return luigi.LocalTarget(f'.cache/folds_{self.num_folds}.h5')

    def run(self):
        self.output().makedirs()

        df = pd.read_csv(self.input().path)

        out_file = h5py.File(self.output().path, 'w')

        train_grp = out_file.create_group('train')
        test_grp = out_file.create_group('test')

        folds = KFold(n_splits=self.num_folds, shuffle=True, random_state=0)

        fold_idx = 0
        for train_idx, test_idx in folds.split(df):
            print(train_idx[:10])
            train_grp.create_dataset(
                f'fold_{fold_idx}',
                train_idx.shape,
                data=train_idx,
            )
            test_grp.create_dataset(
                f'fold_{fold_idx}',
                test_idx.shape,
                data=test_idx,
            )
            fold_idx += 1

        out_file.close()
