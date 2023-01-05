import numpy as np

COLNAMES = [
    "e_r_conversion_rate",
    "conversion_rate_no_treatments",
    "conversion_rate_all_treatments",
    "e_conversion_rate_random",
    "auuc",
    "improvement_to_random",
    "qini_coefficient",
    "euce_k",
    "euce",
    "muce",
    "unique_scores",
    "samples"]


def metrics2row(metrics):
    return [
        metrics.e_r_conversion_rate,
        metrics.conversion_rate_no_treatments,
        metrics.conversion_rate_all_treatments,
        metrics.e_conversion_rate_random,
        metrics.auuc,
        metrics.improvement_to_random,
        metrics.qini_coefficient,
        metrics.k,
        metrics.euce,
        metrics.muce,
        metrics.unique_scores,
        metrics.samples]


def get_training_validation_merged(data):
    training_set = data["training_set"]
    validation_set = data['validation_set']
    training_set['X'] = np.vstack([training_set['X'], validation_set['X']])
    training_set['y'] = np.hstack([training_set['y'], validation_set['y']])
    training_set['t'] = np.hstack([training_set['t'], validation_set['t']])
    training_set['z'] = np.hstack([training_set['z'], validation_set['z']])
    return training_set


def get_training_set(data, i, group_sampling):    
    if i < 0 :
        print("[get_training_set] >>> merging training and validation")
        training_set = get_training_validation_merged(data)
    elif i == 0:
        print("[get_training_set] >>> no undersampling")
        training_set = data["training_set"]
    else:
        print("[get_training_set] >>> undersampling with k=%s group_sampling=%s" % (i, group_sampling))
        training_set = data.k_undersampling(k=i, group_sampling=group_sampling)
    return training_set