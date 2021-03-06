import os
import numpy as np
import h5py


def dump(payload_dict, path):
    try:
        _dump(payload_dict, path)
    except:  # noqa pylint disable=bare-except
        os.remove(path)
        raise


def _handle_str_storage(group_name, data, out_file):
    dtype = h5py.special_dtype(vlen=str)
    out_file.create_dataset(
        group_name,
        dtype=dtype,
        shape=data.shape,
        data=data.astype('S'),
        compression='gzip',
        compression_opts=1,
    )


def _dump(payload_dict, path):
    with h5py.File(path, 'w') as out_file:
        for group_name, data in payload_dict.items():
            if np.issubdtype(data.dtype, np.str):
                _handle_str_storage(group_name, data, out_file)
            else:
                out_file.create_dataset(
                    group_name,
                    dtype=data.dtype,
                    shape=data.shape,
                    data=data,
                    compression='gzip',
                    compression_opts=1,
                )


def load(path, one_or_many_datasets):

    with h5py.File(path, 'r') as in_file:
        if isinstance(one_or_many_datasets, str):
            return in_file[one_or_many_datasets].value

        return tuple([in_file[x].value for x in one_or_many_datasets])
