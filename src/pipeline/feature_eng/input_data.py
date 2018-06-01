import luigi


class TrainSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../input/train.csv')


class TestSet(luigi.ExternalTask):
    def output(self):
        return luigi.LocalTarget('../input/train.csv')
