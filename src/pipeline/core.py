import hashlib
import pandas as pd
import luigi
from sklearn.externals import joblib
# from .h5dataset import dump, load
from .pkl_dataset import dump, load


class PrecomputedFeature(luigi.ExternalTask):
    feature_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f'_features/{self.feature_name}.pkl')


class PrecomputedFold(luigi.ExternalTask):
    fold_id = luigi.IntParameter()

    def output(self):
        return luigi.LocalTarget(f'_folds/fold_{self.fold_id}.h5')


class FilterFeatureToFold(luigi.Task):
    feature_name = luigi.Parameter()
    fold_id = luigi.IntParameter()
    subset = luigi.Parameter()

    id_column = luigi.Parameter()

    def requires(self):
        return {
            'feature': self.clone(PrecomputedFeature),
            'fold': self.clone(PrecomputedFold),
        }

    def output(self):
        return luigi.LocalTarget(f'_feature_folds/{self.feature_name}_{self.fold_id}_{self.subset}.pkl')

    def run(self):
        self.output().makedirs()

        df = joblib.load(self.input()['feature'].path)
        subset_idx = load(self.input()['fold'].path, self.subset)

        subset_df = df.iloc[subset_idx].drop(self.id_column, axis=1)
        joblib.dump(subset_df, self.output().path)


class ComposeDataset(luigi.Task):
    features = luigi.Parameter()
    target = luigi.Parameter()

    fold_id = luigi.IntParameter()
    subset = luigi.Parameter()

    id_column = luigi.Parameter()

    def requires(self):
        features = self.features.split(',')
        return {
            'features': [self.clone(FilterFeatureToFold, feature_name=x) for x in features],
            'target': self.clone(FilterFeatureToFold, feature_name=self.target)
        }

    def output(self):
        hash_content = f'{self.features}|{self.target}|{self.id_column}'
        hash_object = hashlib.md5(hash_content.encode('utf-8'))
        digest = hash_object.hexdigest()[:6]
        return luigi.LocalTarget(f'_feature_folds/combined_{self.fold_id}_{self.subset}_{digest}.pkl')

    def run(self):
        self.output().makedirs()

        dfs = [joblib.load(x.path) for x in self.input()['features']]
        df = pd.concat(dfs, axis=1, join='inner')

        target_df = joblib.load(self.input()['target'].path)

        dump({
            'features': df.values,
            'target': target_df.values,
        },
             self.output().path)
