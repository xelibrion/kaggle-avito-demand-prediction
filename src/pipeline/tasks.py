import pandas as pd
import luigi
from sklearn.externals import joblib

CAT_COLUMNS = [
    'region', 'city', 'parent_category_name', 'category_name', 'param_1', 'param_2',
    'param_3', 'user_type'
]


class TrainSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../input/train.csv')


class TestSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../input/test.csv')


class GetCatValues(luigi.Task):
    category_name = luigi.Parameter()

    def requires(self):
        return {'train': TrainSet(), 'test': TestSet()}

    def output(self):
        cls = self.__class__.__name__.lower()
        return luigi.LocalTarget(f'.cache/{cls}_{self.category_name}.txt')

    def run(self):
        self.output().makedirs()

        df_train = pd.read_csv(self.input()['train'].path)
        df_test = pd.read_csv(self.input()['test'].path)

        train_vals = df_train[self.category_name].dropna().unique()
        test_vals = df_test[self.category_name].dropna().unique()
        vals = set(train_vals).union(set(test_vals))

        with open(self.output().path, 'w') as out_file:
            for val in vals:
                out_file.write(val)
                out_file.write('\n')


class EncodeCategoryTrain(luigi.Task):
    category_name = luigi.Parameter()

    def requires(self):
        return {
            'dataset': TrainSet(),
            'cat_values': GetCatValues(category_name=self.category_name)
        }

    def output(self):
        return luigi.LocalTarget(f'.cache/encoded_train_{self.category_name}.pkl')

    def _cat_values(self):
        with open(self.input()['cat_values'].path) as cat_file:
            return cat_file.readlines()

    def run(self):
        self.output().makedirs()

        df = pd.read_csv(self.input()['dataset'].path)

        df[self.category_name] = pd.Categorical(
            df[self.category_name],
            categories=self._cat_values(),
        )
        df_enc = pd.get_dummies(df[self.category_name])
        joblib.dump(df_enc, self.output().path)


class GetAllCatValues(luigi.Task):
    def requires(self):
        return [EncodeCategoryTrain(category_name=x) for x in CAT_COLUMNS]

    def complete(self):
        return all([x.complete() for x in self.requires()])
