"""
Code for serializing datasets (which takes very long and a lot of memory)
into pickle-format. The point is to do this once, and then let other
tests use much less memory and time.

Run this once for all your experiments (unless you want to randomize the
data again).
"""

import gzip
import pickle
import random
import uplift_modeling.data.load_data as load_data


CRITEO_FORMAT = load_data.DATA_FORMAT
# CRITEO_FORMAT['file_name'] = 'criteo_100k.csv'  # This is for testing.
CRITEO_FORMAT['file_name'] = 'criteo-uplift.csv'  # This is the actual experiment.
# CRITEO_FORMAT['y_idx'] = 14  # for the "visit" label. Using 'conversion' (13) by default.


# HILLSTROM_FORMAT = load_data.HILLSTROM_FORMAT_1
# Already contains filename.

#DATASETS = [HILLSTROM_FORMAT, CRITEO_FORMAT]
DATASETS = [CRITEO_FORMAT]

def create_pickles(path='./datasets/'):
    """
    Function for pickling datasets.

    Args:
    path (str): Path of datasets. Actual file names are
     specified in DATASETS-list above.
    """
    for dataset in DATASETS:
        seed = random.randrange(2**32 - 1)
        dataset['random_seed'] = seed
        # Load data and do preprocessing:
        data = load_data.DatasetCollection(path + dataset['file_name'],
                                           dataset)
        # Write as pickle to disk:
        with gzip.open(path + dataset['file_name'] + str(seed) + '.gz',
                  'wb') as handle:
            pickle.dump(data, handle)


def load_pickle(file_name, path='./datasets/'):
    """
    Function for loading pickled datasets. Zipped or unzipped.

    Args:
    file_name (str): Name of file
    path (str): Location of file
    """
    from data.load_data import DatasetCollection
    # Do gzip version
    fp = gzip.open(path + file_name, 'rb') if file_name.endswith(".gz") else open(path + file_name, 'rb')
    data = pickle.load(fp)
    fp.close()
    #with open(path + file_name, 'rb') as handle:
    #    data = pickle.load(handle)
    return data


if __name__ == '__main__':
    # If main program,
    # create pickle-files.
    create_pickles()
