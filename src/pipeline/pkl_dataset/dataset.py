from sklearn.externals import joblib


def dump(payload_dict, path):
    joblib.dump(payload_dict, path)


def load(path, one_or_many_datasets):
    payload = joblib.load(path)

    if isinstance(one_or_many_datasets, str):
        return payload[one_or_many_datasets]

    return tuple([payload[x] for x in one_or_many_datasets])
