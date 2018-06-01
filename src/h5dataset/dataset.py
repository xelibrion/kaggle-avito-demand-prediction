import h5py


def dump(payload_dict, path):
    with h5py.File(path, 'w') as out_file:
        for group_name, data in payload_dict.items():
            out_file.create_dataset(group_name, shape=data.shape, data=data)


def load(path, one_or_many_datasets):

    with h5py.File(path, 'r') as in_file:
        if isinstance(one_or_many_datasets, str):
            return in_file[one_or_many_datasets].value

        return tuple([in_file[x].value for x in one_or_many_datasets])
