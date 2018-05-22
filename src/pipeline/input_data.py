import os
import pandas as pd
import luigi


class TrainSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../input/train.csv')


class TestSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../input/test.csv')


class WithImagesSet(luigi.Task):
    dataset = luigi.TaskParameter()

    def requires(self):
        return self.dataset

    def output(self):
        out_path = self.input().path.replace('.csv', '_w_images.csv')
        return luigi.LocalTarget(out_path)

    def run(self):
        df = pd.read_csv(self.input().path)
        ft = pd.isnull(df['image'])
        df = df[~ft]
        df.to_csv(self.output().path, index=False)
