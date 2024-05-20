"""
Class for handling datasets for uplift experiments and 
one class to wrap this class in torch-compatible format.

Development ideas:
-Support for ordinal features could perhaps be added.
-Support for continuous dependent variables could perhaps. This is
incompatible with CVT, though.
-Data loading could be made more efficient by discarding observations not belonging
to either treatment group up front and by handling data one observation at a time
rather than reading in all data and handling one feature at a time.
Features cannot be normalized before everything is in memory, but at least
we could reduce the needed memory from 2ND to ND + D or even go as low as ND + 1.
This matters with large datasets. Alternatively use some serialized format
for quick access after preprocessing the dataset once.
-Move DATA_FORMAT and easy access to separate file? Maybe keep it here.
--Implement quick-access functions for the most common datasets.
--Add support to download datasets if not present locally.
-Check whether the LENTA_FORMAT is correct. It probably is not.
-Pytorch dataloaders could also be stored in this object in a dict
in a similar way subsets are stored now.
-This class stores both the original dataset and the subsets. Memory usage?
"""

import csv
import numpy as np
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

# Data format as indices for features, label, and group and
# some other info. This is essentially a configuration.
# This format is used for Criteo-uplift data:
CRITEO_FORMAT = {'file_name': 'criteo-uplift.csv',
               'X_idx': [i for i in range(12)],  # indices for feature columns
               'continuous_idx': [i for i in range(12)],  # indices for continuous features
               'categorical_idx': [],  # indices for categorical indices
               'y_idx': 13,  # index for the dependent variable column
               't_idx': 12,  # index for the treatment label colum
               't_labels': ['1', '0'],  # Treatment and control labels.  
               'random_seed': 1938245,  # Set to None for no randomization.
               # Setting normalization to None implies no normalization.
               # 'v3' centers features and sets variance to 1.
               'normalization': 'v3',
               'headers': True,  # 'True' drops a header row from the data.
               'data_type': 'float32'}  # Data will be set to this type.

# This format is for the Hillstrom Mine that data -challenge dataset
HILLSTROM_FORMAT = {'file_name':
                      'Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv',
                      # Note that many of the features are categorical
                      'X_idx': [i for i in range(8)],
                      # There are a few binary features and a few categorical
                      # ones with multiple values.
                      'continuous_idx': [0, 2],
                      'categorical_idx': [1, 3, 4, 5, 6, 7],
                      # y = 9 corresponds to 'visit'. Conversion has too few positive.
                      'y_idx': 9,  # There is also a possibility to focus on spent money.
                      't_idx': 8,
                      # This format would include men's email as treatment group and
                      # no e-mail as control group.
                      't_labels': ['Mens E-Mail', 'No E-Mail'],  # 'Womens E-Mail'],
                      'random_seed': 6,
                      'headers': True,
                      'normalization': 'v3',
                      'data_type': 'float32'}

# This format is for the Hillstrom Mine that data -challenge dataset
HILLSTROM_CONVERSION = {
    'file_name': 'Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv',
    # Note that many of the features are categorical
    'X_idx': [i for i in range(8)],
    # There are a few binary features and a few categorical
    # ones with multiple values.
    'continuous_idx': [0, 2],
    'categorical_idx': [1, 3, 4, 5, 6, 7],
    # y = 9 corresponds to 'visit'. Conversion has too few positive.
    'y_idx': 10,  # 9 is Visit, 10 is conversion.
    't_idx': 8,
    # This format would include men's email as treatment group and
    # no e-mail as control group.
    't_labels': ['Mens E-Mail', 'No E-Mail'],  # 'Womens E-Mail'],
    'random_seed': 4,  #  6,
    'headers': True,
    'normalization': 'v3',
    'data_type': 'float32'}

# Data available at: https://isps.yale.edu/research/data/d001
# Data looks like:
#sex,yob,g2000,g2002,g2004,p2000,p2002,p2004,treatment,cluster,voted,hh_id,hh_size,numberofnames,p2004_mean,g2004_mean
#"male",1941,"yes","yes","yes","no","yes","No"," Civic Duty",1,"No",1,2,21,.0952381,.8571429
#"female",1947,"yes","yes","yes","no","yes","No"," Civic Duty",1,"No",1,2,21,.0952381,.8571429
VOTER_FORMAT = {'file_name': 'GerberGreenLarimer_APSR_2008_social_pressure.csv',
               'X_idx': [i for i in range(8)] + [12],  # indices for feature columns
               'continuous_idx': [1, 12],  # indices for continuous features
               'categorical_idx': [0, 2, 3, 4, 5, 6, 7],  # indices for categorical indices
               'y_idx': 10,  # index for the dependent variable column
               'y_labels': ['Yes', 'No'],
               't_idx': 8,  # index for the treatment label colum
               't_labels': [' Neighbors', ' Control'],  # #, ' Civic Duty', ' Hawthorne', ' Self'],
               'random_seed': 4,  # 1938245,  # Set to None for no randomization.
               # Setting normalization to None implies no normalization.
               # 'v3' centers features and sets variance to 1.
               'normalization': 'v3',
               'headers': True,  # 'True' drops a header row from the data.
               'data_type': 'float32'}  # Data will be set to this type.


# Finn Kuusisto, Simulated Customer Data 
# In "Support Vector Machines for Differential Prediction"
# stereotypical_customer_simulation.csv
# customer_id,Node1,Node2,Node3,Node4,Node5,Node6,Node7,Node8,Node9,Node10,Node11,Node12,Node13,Node14,Node15,customer_type,Node17,Node18,Node19,Node20,target_control,outcome
# 1,Value4,Value1,Value2,Value2,Value2,Value4,Value2,Value4,Value4,Value4,Value2,Value3,Value2,Value3,Value3,persuadable,Value2,Value2,Value3,Value1,control,negative
CUSTOMER_FORMAT = {'file_name': 'stereotypical_customer_simulation.csv',
               'X_idx': [i for i in range(1, 16)] + [i for i in range(17, 21)],  # indices for feature columns
               'continuous_idx': [],  # indices for continuous features
               'categorical_idx': [i for i in range(1, 16)] + [i for i in range(17, 21)],  # indices for categorical indices
               'y_idx': 22,  # index for the dependent variable column
               'y_labels': ['positive', 'negative'],
               't_idx': 21,  # index for the treatment label colum
               't_labels': ['target', 'control'],
               'random_seed': 1938245,  # Set to None for no randomization.
               # Setting normalization to None implies no normalization.
               # 'v3' centers features and sets variance to 1.
               'normalization': 'v3',
               'headers': True,  # 'True' drops a header row from the data.
               'data_type': 'float32'}  # Data will be set to this type.


# Zenodo data, or pehraps better called Zhao's synthetic data.
# https://zenodo.org/record/3653141#.YwxohPFBw0R
#,trial_id,treatment_group_key,conversion,control_conversion_prob,treatment1_conversion_prob,treatment1_true_effect,x1_informative,x2_informative,x3_informative,x4_informative,x5_informative,x6_informative,x7_informative,x8_informative,x9_informative,x10_informative,x11_irrelevant,x12_irrelevant,x13_irrelevant,x14_irrelevant,x15_irrelevant,x16_irrelevant,x17_irrelevant,x18_irrelevant,x19_irrelevant,x20_irrelevant,x21_irrelevant,x22_irrelevant,x23_irrelevant,x24_irrelevant,x25_irrelevant,x26_irrelevant,x27_irrelevant,x28_irrelevant,x29_irrelevant,x30_irrelevant,x31_uplift_increase,x32_uplift_increase,x33_uplift_increase,x34_uplift_increase,x35_uplift_increase,x36_uplift_increase
# 0,0,control,1,0.5166064824648875,0.572608554747334,0.05600207228244647,-1.9266506898330074,1.2334719980198423,-0.47512021072091604,0.08128310003351527,-2.097355367990696,0.15619370493451765,0.4760174215019512,0.3848542129506569,-1.0666585526632113,-1.168532326602762,1.3167040821139795,-1.0398586967152441,0.717129730558336,0.6141421342662396,0.03910451895611431,-0.6301056161753632,3.161931234783917,0.46475856470886,-0.4726034638576867,-0.18879511623640358,0.46826469708560103,0.9610470189567474,-2.062960044383907,0.41774542328838254,-0.2792847364116334,-1.037129126877536,-0.37814454512454465,-0.11078215707293138,1.0871800027885286,-1.2220691576774803,-0.27900857687902075,1.013911305169857,-0.5708586120652774,-1.158216043140271,-1.3362787988579587,-0.7080559154745507
ZENODO_FORMAT = {'file_name': 'uplift_synthetic_data_100trials.csv',
               'X_idx': [i for i in range(4, 40)],  # indices for feature columns
               'continuous_idx': [i for i in range(4, 40)],  # indices for continuous features
               'categorical_idx': [],  # indices for categorical indices
               'y_idx': 3,  # index for the dependent variable column
               #'y_labels': ['1', '0'],
               't_idx': 2,  # index for the treatment label colum
               't_labels': ['treatment1', 'control'],  # are there other treatments? treatment2 etc.?
               'random_seed': 0,  # originally 1938245,  # Set to None for no randomization.
               # Setting normalization to None implies no normalization.
               # 'v3' centers features and sets variance to 1.
               'normalization': 'v3',
               'headers': True,  # 'True' drops a header row from the data.
               'data_type': 'float32'}  # Data will be set to this type.


# Starbucks format
# ID,Promotion,purchase,V1,V2,V3,V4,V5,V6,V7
# 2,No,0,1,41.3763902,1.17251687,1,1,2,2
# 6,Yes,0,1,25.1635977,0.65305014,2,2,2,2
# 7,Yes,0,1,26.5537781,-1.5979723,2,3,4,2
# 10,No,0,2,28.5296907,-1.0785056,2,3,2,2
STARBUCKS_FORMAT = {'file_name': 'starbucks.csv',
               'X_idx': [i for i in range(3, 10)],  # indices for feature columns
               'continuous_idx': [i for i in range(3, 10)],  # indices for continuous features, not sure whether features are actually continuous...
               'categorical_idx': [],  # indices for categorical indices
               'y_idx': 2,  # index for the dependent variable column
               #'y_labels': ['1', '0'],
               't_idx': 1,  # index for the treatment label colum
               't_labels': ['Yes', 'No'],  # are there other treatments? treatment2 etc.?
               'random_seed': 4,  # 1938245,  # Set to None for no randomization.
               # Setting normalization to None implies no normalization.
               # 'v3' centers features and sets variance to 1.
               'normalization': 'v3',
               'headers': True,  # 'True' drops a header row from the data.
               'data_type': 'float32'}  # Data will be set to this type.


# Only the y_idx is set for Lenta-format so far. Everything else needs to be done (now copies of Starbuck's).
LENTA_FORMAT = {'file_name': 'lenta_dataset.csv',
               'X_idx': [i for i in range(3, 10)],  # indices for feature columns
               'continuous_idx': [i for i in range(3, 10)],  # indices for continuous features, not sure whether features are actually continuous...
               'categorical_idx': [],  # indices for categorical indices
               'y_idx': 156,  # index for the dependent variable column, continuous outcome at [2].
               #'y_labels': ['1', '0'],
               't_idx': 56,  # index for the treatment label colum
               't_labels': ['Yes', 'No'],  # are there other treatments? treatment2 etc.?
               'random_seed': 1938245,  # Set to None for no randomization.
               # Setting normalization to None implies no normalization.
               # 'v3' centers features and sets variance to 1.
               'normalization': 'v3',
               'headers': True,  # 'True' drops a header row from the data.
               'data_type': 'float32'}  # Data will be set to this type.


class DatasetCollection(object):
    """
    Class for handling datasets for uplift modeling.
    This class provides easy access to the features, variables,
    and variable transformations of a dataset, and easy subsetting
    into relevant subsets for uplift modeling.

    At initialization, the class loads a dataset from a csv-file, preprocesses
    it (normalization, one-hot encoding, variable transformations), and
    subsamples the dataset into training, validation, and testing sets.

    This class contains **many undersampling methods,** including k-undersampling from Nyberg &
    Klami (2021), split undersampling from Nyberg & Klami (2023), and naive undersampling
    from Nyberg & Klami (2021) (undoing the effects of undersampling, i.e.
    calibration of the uplift predictions, needs to be done separately).
    There is also a method for reducing the dataset size
    by random sampling that overwrites the training and validation set in the object.
    This can be useful to study performance of a model over dataset size.

    Parameters
    ----------
    file_name : str
        Name of the csv-file containing the dataset with relative or absolute path.
    data_format : dict
        A dictionary containing necessary details of the dataset. See example below
        for the Voter-dataset (Gerber, Green, & Larimer, 2008).

    .. code-block::

        data_format = {'file_name': 'GerberGreenLarimer_APSR_2008_social_pressure.csv',
        'X_idx': [i for i in range(8)] + [12],  # Indices for columns with features
        'continuous_idx': [1, 12],  # Indices for continuous features
        'categorical_idx': [0, 2, 3, 4, 5, 6, 7],  # Indices for categorical features
        'y_idx': 10,  # Index for the column containing the dependent variable
        'y_labels': ['Yes', 'No'],  # If provided, the first item in the list is converted
        # to the positive label and the second to the negative. Otherwise the contents
        # of the y_idx-column is converted by bool().
        't_idx': 8,  # Index for the treatment label colum
        't_labels': [' Neighbors', ' Control'],  # Labels for the selected treatments.
        # Two treatment labels have to be provided.
        'random_seed': 4,  # Seed for data shuffling. Set to None for no randomization.
        'normalization': 'v3',  # 'v3' results in centering and normalization to unit variance
        # 'v2' results in unit variance without centering, and None keeps the data as is.
        # of continuous features.
        'headers': True,  # 'True' drops a header row from the data.
        'data_type': 'float32'}  # Data will be set to this type.
    
    Accessing the data
    ------
    The features of the training set can be accessed with ``tmp_data['training_set']['X']``,
    the treatment labels with ``tmp_data['training_set']['t']``, the dependent variable with
    ``tmp_data['training_set']['y']``, the class-variable transformation with 
    ``tmp_data['training_set']['z']`` (Jaskowski & Jaroszewicz, 2012), and the
    outcome-transformation label with ``tmp_data['training_set']['r']`` (Athey & Imbens, 2016). 

    Three subsets are prepared at initialization: the ``'training_set'``, the ``'validation_set'``,
    and the ``'testing_set'``. These can be accessed following the pattern above. In addition,
    the treated and untreated observations can be accessed with 
    ``tmp_data['testing_set', 'treatment']['X']`` and 
    ``tmp_data['validation_set', 'control']['X']``.
    
    Note that the CVT-transformation ``'z'`` does not automatically subsample the observations
    so that there are equally many treated and untreated observations. CVT requires that
    p(t=1) = p(t=0). This can be implemented by weighting, or by subsampling. The subsampling
    is implemented in this class and can be accessed with the ``'one_to_one'``-flag, e.g.
    ``tmp_data['training_set', 'all', 'one_to_one']['X']``.
    """
    def __init__(self, file_name, data_format):
        self.file_name = file_name
        self.data_format = data_format
        self.nro_features = len(self.data_format['X_idx'])
        # Load data into array. All data is in csv or similar format.
        # Load data into self.X, self.y, self.t and self.z:
        # Do some light preprocessing (e.g. identify class label, features and group).
        # This function will parse the data into X, y, and t following the format
        # specified in self.data_Format.
        self._load_data()
        # Randomize
        self._shuffle_data()

        # Create empty dict for storing of usable datasets such as training set etc:
        self.datasets = {}
        # Populate self.datasets with predefined subsets:
        self._create_subsets()

    def _load_data(self):
        """
        Method for loading data from file and parsing it following the
        data_format provided in the initializer.
        Currently supports csv-files.
        """
        tmp_data = []
        if self.file_name.endswith('csv'):
            try:
                with open(self.file_name, "r") as handle:
                    file_ = csv.reader(handle)
                    for row in file_:
                        tmp_data.append(row)
            except FileNotFoundError:
                raise Exception("The file {} was not found in the working directory.".format(self.file_name))
            if self.data_format['headers']:
                # Drop title row:
                tmp_data = tmp_data[1:]
        else:
            raise Exception("The file needs to be a .csv-file.")

        # Parse data into suitable format following format specified in
        # self.data_format:
        tmp_X = None
        for col, _ in enumerate(tmp_data[0]):
            # Parse one column at a time into a suitable format.
            # Columns contain features, treatment labels, and outcome labels.
            # If column is feature:
            if col in self.data_format['X_idx']:
                if col in self.data_format['continuous_idx']:
                    # Column contains a continuous feature
                    tmp = np.array([[row[col]] for row in tmp_data])
                    tmp = tmp.astype(self.data_format['data_type'])
                    tmp = self._normalize_data(tmp)
                elif col in self.data_format['categorical_idx']:
                    # Column contains a categorical feature
                    tmp_col = [row[col] for row in tmp_data]
                    keys = np.unique(tmp_col)
                    tmp = np.zeros([len(tmp_col), len(keys)], dtype=
                                   self.data_format['data_type'])
                    # One-hot encoding for categorical features:
                    for i, key in enumerate(keys):
                        for j, _ in enumerate(tmp_col):
                            if tmp_col[j] == key:
                                tmp[j, i] = 1
                # Add new features to tmp_X
                if tmp_X is None:
                    tmp_X = tmp
                else:
                    tmp_X = np.concatenate([tmp_X, tmp], axis = 1)
            elif col == self.data_format['t_idx']:
                # Column contains the treatment label.
                # Seems that the format only checks whether
                # Create binary group label for _every_ observation
                # where True indicates treatment group.
                # Observations not beloning to either treatment group
                # are filtered out later.
                tmp = [row[col] for row in tmp_data]
                tmp_t = np.array([item == self.data_format['t_labels'][0] for
                                  item in tmp])
            elif col == self.data_format['y_idx']:
                if 'y_labels' in self.data_format.keys():
                    # If labels are given for the class, the first label in
                    # data_format['y_labels'] is considered the positive label:
                    tmp = [row[col] for row in tmp_data]
                    tmp_y = np.array([item == self.data_format['y_labels'][0] for
                                      item in tmp])
                else:
                    # No 'y_label' is specified, reading in as boolean.
                    tmp_y = np.array([row[col] for row in tmp_data])
                    tmp_y = tmp_y.astype(np.bool_)
        # Class-variable transformation:
        tmp_z = tmp_y == tmp_t  # CVT
        tmp_r = self._revert_label(tmp_y, tmp_t)  # OTM transformation

        # Keep only observations that belong to treatment groups
        # specified in data_format['t_labels']:
        tmp = [row[self.data_format['t_idx']] for row in tmp_data]
        group_idx = [item in self.data_format['t_labels'] for item in
                     tmp]
        group_idx = np.array(group_idx)
        self.X = tmp_X[group_idx, :]
        self.y = tmp_y[group_idx]
        self.t = tmp_t[group_idx]
        self.z = tmp_z[group_idx]
        self.r = tmp_r[group_idx]


        # Print some statistics for the loaded data:
        print("Dataset {} loaded".format(self.file_name))
        print("\t\t\t#y\t#samples\tconversion rate")
        print("All samples", end='\t\t')
        print(sum(self.y), end='\t')
        print(len(self.y), end='\t')
        print(sum(self.y)/len(self.y))
        print("Treatment samples", end='\t')
        print(sum(self.y[self.t]), end='\t')
        print(sum(self.t), end='\t')
        print(sum(self.y[self.t]/sum(self.t)))
        print("Control samples", end='\t\t')
        print(sum(self.y[~self.t]), end='\t')
        print(sum(~self.t), end='\t')
        print(sum(self.y[~self.t]/sum(~self.t)))
        conversion_rate_treatment = sum(self.y[self.t])/sum(self.t)
        conversion_rate_control = sum(self.y[~self.t])/sum(~self.t)
        effect_size = (conversion_rate_treatment - conversion_rate_control) /\
                      conversion_rate_control
        print("Estimated effect size for treatment: {:.2f}% (percentage points)".format(effect_size * 100))


    def _shuffle_data(self):
        if self.data_format['random_seed'] is not None:
            print("Random seed set to {} and data shuffled.".format(self.data_format['random_seed']))
            # Set random seed to get same split for all experiments
            np.random.seed(self.data_format['random_seed'])
        shuffling_idx = np.random.choice([item for item in range(len(self.y))],
                                         len(self.y), replace=False)
        self.X = self.X[shuffling_idx, :]
        self.y = self.y[shuffling_idx]
        self.t = self.t[shuffling_idx]
        self.z = self.z[shuffling_idx]
        self.r = self.r[shuffling_idx]

    def _create_subsets(self, mode='basic'):
        """
        Method for creating basic training, validation, and testing sets.
        Method further creates an additional training_set_2, validation_set_2a, 
        and validation_set_2b to be used for deciding on early stopping when 
        training neural networks.
        Training, validation, and testing sets are split 50:25:25, and
        training2, validation_set_2a, validation_set_2b, and testing set are
        split 6/16:3/16:3/16:4/16. Note that in both cases, the testing sets are
        identical in all aspects.

        Parameters
        ----------
        mode : str
            Mode defines which datasets are added. 'basic' adds training, testing,
            and validation sets.
        """
        # Using a 50/25/25 split
        n_samples = self.X.shape[0]
        if mode == 'basic':
            # Add basic training, validation, and testing sets to self.datasets:
            self.add_set('training_set', 0, n_samples // 2)
            self.add_set('validation_set', n_samples // 2, n_samples * 3 // 4)
            self.add_set('testing_set', n_samples * 3 // 4, n_samples)
        elif mode == 'one_to_one':
            # Subsampled training set with 1:1 ratio of treated and untreated observations.
            # Useful for class-variable transformation.
            tmp = self._subsample_one_to_one('training_set')
            self.datasets.update({'training_set_11': tmp})
            # if 'training_set_2' in self.datasets.keys():
            #     # I see no use for this.
            #     # Add this only if the second training set split is in use:
            #     tmp = self._subsample_one_to_one('training_set_2')
            #     self.datasets.update({'training_set_2_11': tmp})
        elif mode == 'two_validation_sets':
            # Add also slightly different split that enables early stopping of
            # neural networks using separate validation set.
            # Idx n_samples * 12 // 16 to n_samples * 16 // 16 is the testing set.
            self.add_set('training_set_2', 0, n_samples * 6 // 16)
            self.add_set('validation_set_2a', n_samples * 6 // 16, n_samples * 9 // 16)
            self.add_set('validation_set_2b', n_samples * 9 // 16, n_samples * 12 // 16)
        else:
            print("Mode '{}' not recognised.".format(mode))

    def _revert_label(self, y_vec, t_vec):
        """
        Variable transformation ("outcome transformation method," "revert-label") 
        approach following Athey & Imbens (2015).

        Parameters:
        -----------
        y_vec : np.array([float])) 
            Array of conversion values in samples.
        t_vec : np.array([bool]) 
            Array of treatment labels for same samples.
        """
        N_t = sum(t_vec == True)
        N_c = sum(t_vec == False)
        # Sanity check:
        assert N_t + N_c == len(t_vec), "Error in sample count (_revet_label())."
        # This needs to be sorted out.
        p_t = N_t / (N_t + N_c)
        assert 0.0 < p_t < 1.0, "Revert-label cannot be estimated from only t or c observations."
        def revert(y_i, t_i, p_t):
            return (y_i * (int(t_i) - p_t) / (p_t * (1 - p_t)))
        r_vec = np.array([revert(y_i, t_i, p_t) for y_i, t_i in zip(y_vec, t_vec)])
        r_vec = r_vec.astype(self.data_format['data_type'])
        return r_vec

    def add_set(self, name, start_idx, stop_idx):
        """
        This method adds a subset of the entire dataset to the object as
        a subset to be accessed with the name specified, e.g. 
        ``tmp_data['calibration_set']['X']`` if the name was set to
        ``'calibration_set'``.

        Parameters
        ----------
        name : str
            Name of the subset. The dataset can later be accessed using this key.
        start_idx : int
            Index of the first observation to be included in the subset. start_idx 
            and stop_idx together defines a range.
        stop_idx : int
            Index of the first observation to **not be included** in the subset.
        """
        X_tmp = self.X[start_idx:stop_idx, :]
        y_tmp = self.y[start_idx:stop_idx]
        t_tmp = self.t[start_idx:stop_idx]
        z_tmp = y_tmp == t_tmp
        r_tmp = self._revert_label(y_tmp, t_tmp)
        self.datasets.update({name: {'X': X_tmp,
                                     'y': y_tmp,
                                     't': t_tmp,
                                     'z': z_tmp,
                                     'r': r_tmp}})


    def _normalize_data(self, vector):
        """
        Method for normalizing data. Used at initialization by _load_data().

        Parameters:
        -----------
        (data_format['normalization'] : str)
            'normalization' needs to be passed in data_format at initialization.
            None to keep data as is, 'v1' for normalization of vector over all users of one feature
            to unit variance (not recommended), 'v2' for unit variance, 'v3' for centralization and
            unit variance. 'v3' or None is recommended.
        """
        if self.data_format['normalization'] is None:
            pass
        elif self.data_format['normalization'] == 'v1':
            # This sets the length of the vector representing one feature over all
            # observations to 1. Not great.
            tmp = normalize(vector)
        elif self.data_format['normalization'] == 'v2':
            # Set variance to 1
            tmp = vector / np.std(vector)
        elif self.data_format['normalization'] == 'v3':
            # This normalizes features as stated in Diemert & al.
            # Set mean to 0 and variance to 1:
            tmp = vector - np.mean(vector)
            tmp = tmp / np.std(tmp)
        return tmp

    def _subsample_one_to_one(self, target_set):
        """
        Method to further subsample some subset (training/validation/testing 
        set) so that the ratio between treated and untreated observations is 
        1:1. Method returns the subsampled data and leaves self as is.

        Parameters:
        -----------
        target_set : str
            Name of set to be subsampled. Must exist in self.datasets.keys().
        """
        tmp = self.datasets[target_set]
        tot_samples = len(tmp['t'])
        # Smaller of these is the number of samples we want
        n_samples = int(np.min([sum(tmp['t']), sum(~tmp['t'])]))
        treatment_idx = np.array([i for i, t in zip(range(tot_samples), tmp['t'])
                                    if t])
        control_idx = np.array([i for i, t in zip(range(tot_samples), tmp['t'])
                                if not t])
        # Shuffle:
        np.random.shuffle(treatment_idx)
        np.random.shuffle(control_idx)
        idx = np.concatenate([treatment_idx[:n_samples],
                                control_idx[:n_samples]])
        # Shuffle index for good measure (prevent any idiosyncrasies in
        # algorithms to have weird effects)
        np.random.shuffle(idx)
        X = tmp['X'][idx, :]
        y = tmp['y'][idx]
        t = tmp['t'][idx]
        z = tmp['z'][idx]
        r = self._revert_label(y, t)
        # Revert-label does not make much sense together with undersampling.
        # (techically it is possible to calculate, but the normalization is
        # precisely there to make undersampling unnecessary.)
        # At minimum, it would need to be recalculated for a subset.
        return {'X': X, 'y': y, 't': t, 'z': z, 'r': r}


    def k_undersampling(self, k, group_sampling='natural'):
        """
        Method returns a training set where the rate of positive observations
        is changed by a factor of k within both the treated and the untreated
        subsets so that :math:`p^*(y=1|t=1) = k \cdot p(y=1|t=1)` and 
        :math:`p^*(y=1|t=0) = k \cdot p(y=1|t=0)` where :math:`p^*(y=1|t)` is the positive rate
        after undersmapling by either reducing the number of
        negative observations (k>1) or reducing the number of positive 
        observations (k<1).
        The method can also change the sampling rate of treatment vs. control
        observations to 1:1.
        This is the original implementation of k-undersampling by
        Nyberg & Klami 2021.

        Parameters
        ----------
        k : int
            This number will determine the change in positive rate in the data.
        group_sampling : str
            'natural' implies no change in group sampling rate, i.e. the number
            of observations in the treatment and control groups stay constant.
            '1:1' indicates that there should be equally many treatment and
            control observations. This is useful with CVT and enforces
            :math:`p(t=0) = p(t=1)`.
        """
        # Number of positives in treatment group:
        t_data = self['training_set', 'treatment']
        num_pos_t = sum(t_data['y'])
        # Find indices for all positive treatment samples
        pos_idx_t = np.array([i for i, tmp in enumerate(zip(self['training_set']['y'],
                                                            self['training_set']['t']))
                              if bool(tmp[0]) is True and bool(tmp[1]) is True])
        num_neg_t = sum(~t_data['y'])
        # Find indices for all negative treatment samples:
        neg_idx_t = np.array([i for i, tmp  in enumerate(zip(self['training_set']['y'],
                                                             self['training_set']['t']))
                              if bool(tmp[0]) is False and bool(tmp[1]) is True])
        num_tot_t = len(t_data['y'])

        c_data = self['training_set', 'control']
        num_pos_c = sum(c_data['y'])
        # Find indices for all positive control samples
        pos_idx_c = np.array([i for i, tmp in enumerate(zip(self['training_set']['y'],
                                                            self['training_set']['t']))
                              if bool(tmp[0]) is True and bool(tmp[1]) is False])
        num_neg_c = sum(~c_data['y'])
        # Find indices for all negative control samples:
        neg_idx_c = np.array([i for i, tmp in enumerate(zip(self['training_set']['y'],
                                                            self['training_set']['t']))
                              if bool(tmp[0]) is False and bool(tmp[1]) is False])
        num_tot_c = len(c_data['y'])
        # Adjust the total number of positive and negative samples in treatment and
        # control groups separately to change the positive rate by k:
        if k >= 1:
            num_neg_c_new = int(round(num_tot_c // k) - num_pos_c)
            assert num_neg_c_new > 0, "k {} causes all negative control samples to be dropped".format(k)
            num_neg_t_new = int(round(num_tot_t // k) - num_pos_t)
            assert num_neg_t_new > 0, "k {} causes all negative treatment samples to be dropped".format(k)
            num_pos_c_new = num_pos_c
            num_pos_t_new = num_pos_t
        elif k < 1 and k > 0:  # Aiming for k * pos/tot = pos_new / tot_new
            num_neg_c_new = num_neg_c  # stays constant
            num_pos_c_new = int(round(k * num_pos_c / num_tot_c * num_neg_c /\
                (1 - k * num_pos_c / num_tot_c)))
            assert num_pos_c_new > 0, "k {} causes all positive control samples to be dropped".format(k)
            num_neg_t_new = num_neg_t  # stays constant
            num_pos_t_new = int(round(k * num_pos_t / num_tot_t * num_neg_t /\
                (1 - k * num_pos_t / num_tot_t)))
            assert num_pos_t_new > 0, "k {} causes all positive control samples to be dropped".format(k)
        else:
            raise ValueError("k needs to be larger than 0")

        # Change number of samples to be picked in treatment or control group to
        # make num_tot_t == num_tot_c:
        if group_sampling == '1:1':
            num_tot_c_new = num_neg_c_new + num_pos_c_new
            num_tot_t_new = num_neg_t_new + num_pos_t_new
            if num_tot_c_new > num_tot_t_new:
                # Reduce number of control samples:
                coef = num_tot_t_new / num_tot_c_new
                num_neg_c_new = int(coef * num_neg_c_new)
                num_pos_c_new = int(coef * num_pos_c_new)
            else:
                # Reduce number of treatment samples:
                coef = num_tot_c_new / num_tot_t_new
                num_neg_t_new = int(coef * num_neg_t_new)
                num_pos_t_new = int(coef * num_pos_t_new)

        # Create indices for sampling:
        new_neg_c_idx = np.random.choice(neg_idx_c, size=num_neg_c_new, replace=False)
        new_neg_t_idx = np.random.choice(neg_idx_t, size=num_neg_t_new, replace=False)
        new_pos_c_idx = np.random.choice(pos_idx_c, size=num_pos_c_new, replace=False)
        new_pos_t_idx = np.random.choice(pos_idx_t, size=num_pos_t_new, replace=False)

        idx = np.concatenate([new_pos_t_idx, new_neg_t_idx,
                              new_pos_c_idx, new_neg_c_idx],
                             axis=0)
        # Shuffle in place for good measure
        np.random.shuffle(idx)
        tmp_X = self['training_set']['X'][idx, :]
        tmp_y = self['training_set']['y'][idx]
        tmp_t = self['training_set']['t'][idx]
        tmp_z = self['training_set']['z'][idx]
        # We might also need 'r' here (for estimation of E(MSE)-gradient)
        tmp_r = self._revert_label(tmp_y, tmp_t)

        return {'X': tmp_X, 'y': tmp_y, 'z': tmp_z, 't': tmp_t, 'r': tmp_r}


    def split_undersampling(self, k_t, k_c=None, group_sampling=None,
                            seed=None, target_set='training_set'):
        """
        Method to undersample the training set. The undersampling 
        is performed so that p(y=1) in the original data equals 
        p(y=1) / k_t (or k_c) in the undersampled data for treated 
        and control samples separately. If k_c is not provided,
        and group_sampling is set to '11', the behavior is
        identical to k_undersampling save for randomization.
        This is the original implementation for split undersampling
        from Nyberg & Klami 2023.

        Parameters
        -----------
        k_t : float 
            Value of k for undersampling of treated samples.
            Value in [0, inf]. For values below 1, positive samples are
            dropped.
        k_c : float 
            Gets value k_t unless k_c is specified. I that case
            it works like k_undersampling.
        group_sampling : str 
            If set to '11' there will be equally many
            treatment and control samples.
        seed : int 
            Random seed for numpy. If set to None, random seed is
            not set.
        target_set : str 
            Which set to sample the undersampled dataset from. Needs
            to exist as key in DatasetCollection.

        Returns
        -------
        dict
            Data undersampled with desired properties is returned in a
            dict with keys 'X', 'y', 't', 'z', and 'r'.
        """
        if k_c is None:
            # Use same k for both treated and control samples.
            k_c = k_t

        # Number of positives in treatment group:
        t_data = self[target_set, 'treatment']
        num_pos_t = sum(t_data['y'])
        # Find indices for all positive treatment samples
        pos_idx_t = np.array([i for i, tmp in enumerate(zip(self[target_set]['y'],
                                                            self[target_set]['t']))
                              if bool(tmp[0]) is True and bool(tmp[1]) is True])
        num_neg_t = sum(~t_data['y'])
        # Find indices for all negative treatment samples:
        neg_idx_t = np.array([i for i, tmp  in enumerate(zip(self[target_set]['y'],
                                                             self[target_set]['t']))
                              if bool(tmp[0]) is False and bool(tmp[1]) is True])
        num_tot_t = len(t_data['y'])
        assert k_t * num_pos_t / num_tot_t < 1, "Not enough negative treatment samples for k_t: {}".format(k_t)

        c_data = self[target_set, 'control']
        num_pos_c = sum(c_data['y'])
        # Find indices for all positive control samples
        pos_idx_c = np.array([i for i, tmp in enumerate(zip(self[target_set]['y'],
                                                            self[target_set]['t']))
                              if bool(tmp[0]) is True and bool(tmp[1]) is False])
        num_neg_c = sum(~c_data['y'])
        # Find indices for all negative control samples:
        neg_idx_c = np.array([i for i, tmp in enumerate(zip(self[target_set]['y'],
                                                            self[target_set]['t']))
                              if bool(tmp[0]) is False and bool(tmp[1]) is False])
        num_tot_c = len(c_data['y'])
        assert k_c * num_pos_c / num_tot_c < 1, "Not enough negative samples for k_c: {}".format(k_c)
        # Adjust the total number of positive and negative samples in treatment and
        # control groups separately to change the positive rate by k:

        if k_t >= 1:
            # Drop negative samples for k >= 1:
            num_neg_t_new = max(0, int(num_tot_t / k_t) - num_pos_t)
            num_pos_t_new = num_pos_t
        elif 1 > k_t > 0:  # Aiming for k * pos/tot = pos_new / tot_new
            # Drop positive samples for k < 1:
            num_neg_t_new = num_neg_t  # stays constant
            num_pos_t_new = max(0, int(k_t * num_pos_t / num_tot_t * num_neg_t /\
                                       (1 - k_t * num_pos_t / num_tot_t)))
        else:
            raise ValueError("k_t needs to be larger than 0")

        if k_c >= 1:
            # Drop negative samples for k >= 1:
            num_neg_c_new = max(0, int(num_tot_c / k_c) - num_pos_c)
            num_pos_c_new = num_pos_c
        elif 1 > k_c > 0:  # Aiming for k * pos/tot = pos_new / tot_new
            # Drop positive samples for k < 1:
            num_neg_c_new = num_neg_c  # stays constant
            num_pos_c_new = max(0, int(k_c * num_pos_c / num_tot_c * num_neg_c /\
                                       (1 - k_c * num_pos_c / num_tot_c)))
        else:
            raise ValueError("k_c needs to be larger than 0")

        # Change number of samples to be picked in treatment or control group to
        # make num_tot_t == num_tot_c:
        if group_sampling == '11':
            num_tot_c_new = num_neg_c_new + num_pos_c_new
            num_tot_t_new = num_neg_t_new + num_pos_t_new
            if num_tot_c_new > num_tot_t_new:
                # Reduce number of control samples:
                coef = num_tot_t_new / num_tot_c_new
                num_neg_c_new = int(coef * num_neg_c_new)
                num_pos_c_new = int(coef * num_pos_c_new)
            else:
                # Reduce number of treatment samples:
                coef = num_tot_c_new / num_tot_t_new
                num_neg_t_new = int(coef * num_neg_t_new)
                num_pos_t_new = int(coef * num_pos_t_new)

        # Create indices for sampling:
        new_neg_c_idx = np.random.choice(neg_idx_c, size=num_neg_c_new, replace=False)
        new_neg_t_idx = np.random.choice(neg_idx_t, size=num_neg_t_new, replace=False)
        new_pos_c_idx = np.random.choice(pos_idx_c, size=num_pos_c_new, replace=False)
        new_pos_t_idx = np.random.choice(pos_idx_t, size=num_pos_t_new, replace=False)

        idx = np.concatenate([new_pos_t_idx, new_neg_t_idx,
                              new_pos_c_idx, new_neg_c_idx],
                             axis=0)
        # Shuffle in place for good measure
        if seed is not None:
            # Set random seed.
            # Using a separate random number generator not to mess
            # with the global RNG of Numpy.
            rng = np.random.default_rng(seed)
            rng.shuffle(idx)
        else:
            # Do as was done before the change with random number
            # generators in Numpy (1.20.1?).
            np.random.shuffle(idx)
        tmp_X = self['training_set']['X'][idx, :]
        tmp_y = self['training_set']['y'][idx]
        tmp_t = self['training_set']['t'][idx]
        tmp_z = self['training_set']['z'][idx]
        # We will also need 'r' here now (MSE-gradient)!
        tmp_r = self._revert_label(tmp_y, tmp_t)

        return {'X': tmp_X, 'y': tmp_y, 'z': tmp_z, 't': tmp_t, 'r': tmp_r}

    def naive_undersampling(self, k):
        """
        Naive version of undersampling dropping either negative or positive
        observations from the treated and untreated subsets of the training
        set with equal probability so that :math:`p(y=1) = k \cdot p^*(y=1)`
        where :math:`p^*(y=1)` is the positive rate after undersampling.
        This is the original implementation from Nyberg & Klami (2021).

        Parameters
        ----------
        k : float
            Undersampling factor. In ]0, inf], although upper boundary naturally comes
            from number of observations that can be dropped before dropping *all* majority
            class observations. k > 1 will lead to negative observations being dropped and k < 1
            to positive ones being dropped.

        Returns
        -------
        dict
            A dictionary with the subsetted data with keys 'X', 'y', 't', 'z', 'r'.
        """
        # 1. Estimate what k_t and k_c values a common k would correspond to 
        # 2. Call split_undersampling.
        n_samples = len(self['training_set', 'all']['y'])
        n_positives = sum(self['training_set', 'all']['y'])
        n_negatives = n_samples - n_positives
        conversion_rate = n_positives / n_samples
        assert k > 0, "k needs to be larger than 0."
        assert k * conversion_rate <= 1, "Given k too large for data."
        if k >= 1:
            # k > 1 implies negative samples are dropped.
            # Number of negative samples to be dropped:
            # (again assuming we are dropping negative samples)
            n_neg_drop = n_samples * (1 - 1 / k)  # This when k >= 1.
            drop_rate = n_neg_drop / n_negatives
            # Treated samples:
            tmp_pos_n = sum(self['training_set', 'treatment']['y'])
            tmp_n = len(self['training_set', 'treatment']['y'])
            tmp_samples_to_drop = (tmp_n - tmp_pos_n) * drop_rate
            k_t = (tmp_pos_n / (tmp_n - tmp_samples_to_drop)) / (tmp_pos_n / tmp_n)
            # Control samples:
            tmp_pos_n = sum(self['training_set', 'control']['y'])
            tmp_n = len(self['training_set', 'control']['y'])
            tmp_samples_to_drop = (tmp_n - tmp_pos_n) * drop_rate
            k_c = (tmp_pos_n / (tmp_n - tmp_samples_to_drop)) / (tmp_pos_n / tmp_n)
        elif 0 < k < 1:
            n_pos_drop = n_positives - k * n_positives * n_negatives /\
                (n_samples - k * n_positives)
            drop_rate = n_pos_drop / n_positives
            print("n_pos_drop: {}".format(n_pos_drop))
            print("n_samples: {}".format(n_samples))
            print("k: {}".format(k))
            print("n_positives: {}".format(n_positives))
            print("n_negatives: {}".format(n_negatives))
            print("Drop rate: {}".format(drop_rate))
            # Treated samples:
            tmp_pos_n = sum(self['training_set', 'treatment']['y'])
            tmp_n = len(self['training_set', 'treatment']['y'])
            tmp_samples_to_drop = tmp_pos_n * drop_rate
            k_t = (tmp_pos_n - tmp_samples_to_drop) /\
                (tmp_n - tmp_samples_to_drop) /\
                    (tmp_pos_n / tmp_n)
            # Control samples:
            tmp_pos_n = sum(self['training_set', 'control']['y'])
            tmp_n = len(self['training_set', 'control']['y'])
            tmp_samples_to_drop = tmp_pos_n * drop_rate
            k_c = (tmp_pos_n - tmp_samples_to_drop) /\
                (tmp_n - tmp_samples_to_drop) /\
                    (tmp_pos_n / tmp_n)
        else:
            # Should never end up here.
            raise ValueError("k not valid.")
        return self.split_undersampling(k_t=k_t, k_c=k_c)


    def __getitem__(self, *args):
        """
        Shorthand method to access self.datasets' contents. This function will
        return either a numpy-array or a pytorch dataloader depending on parameters.

        Parameters
        ----------
        args[0] : str
            name of key in self.datasets to access, e.g. 'training_set',
            'validation_set', or 'testing_set'. Any name of a subset
            added to self.dataset-dict is valid.
        args[1] : str
            Optional. 'all', 'treatment', or 'control' to get subset with
            all, treated, or untreated ("control") observations. Leaving
            it empty will behave same as 'all'.
        args[2] : str
            Optional. If set to 'one_to_one', the getter will return data
            subsetted so that the ratio between treated and untreated
            observations is 1:1.


        args[1] = undersampling {None, '11', '1111'}: None causes no undersampling
         at all, '11' results in treatment and control groups being equally large,
         '1111' results in '11' and #positive and #negative in both groups to be
         equally large.
        args[2] = group {'all', 'treatment', 'control'}: 'all' and None both
         return all data. 'treatment' returns samples that were treated etc.

        Returns
        -------
        dict
            Dict with appropriate data with the customary keys.

        Notes
        -Fix overall structure.
        -Must check for 
        -If 'one_to_one' dataset does not exist in self.datasets.keys(), create
         the dataset.
        """
        # Handle input arguments:
        group = 'all'  # Subset for treatment or control?
        if isinstance(args[0], str):  
            # Only a name of a subset was passed, no more parameters to be expected.
            target_set = args[0]
        elif isinstance(args[0], tuple):  # Multiple arguments were passed
            target_set = args[0][0]
            # Checking for existence of subset:
            if len(args[0]) > 1:  # Still more arguments
                group = args[0][1]
                if len(args[0]) > 2:  # There are still arguments left
                    one_to_one = args[0][2]
                    if one_to_one == 'one_to_one':
                        # A subset is requested where the ratio between treated and untreated
                        # observations is 1:1.
                        # Check existence:
                        if target_set == 'training_set':
                            if target_set + '11' not in self.datasets.keys():
                                # If it does not exist, create it.
                                self._create_subsets(mode='one_to_one')
                                target_set = target_set + '_11'
                        else:
                            raise Exception("One-to-one subsampled dataset only available for training set.")
                            print("Currently no undersampled datasets for other " +
                                "than training set.")
                    if len(args[0]) > 3:
                        raise Exception("Too many arguments.")

        if target_set not in self.datasets.keys():
            # Target_set does not exist yet.
            if target_set in ['training_set_2', 'validation_set_2a',
                                'validation_set_2b']:
                txt = "Creating three new subsets, 'training_set_2', 'validation_set_2a' \n"
                + "and 'validation_set_2b'. These are suitable for early stopping. \n"
                + "6/16 is for the training set, 3/16 for the first validation set, \n"
                + "and 3/16 for the second validation set \n"
                + "The testing set (4/16 of the data) remains unchanged."
                print(txt)
                self._create_subsets(mode='two_validation_sets')
            else:
                txt = "No dataset named {} available.".format(target_set)
                raise Exception(txt)
        # Store approproate data in tmp:
        if group == 'treatment':
            idx = self.datasets[target_set]['t']
            tmp = self.subset_by_group(target_set, idx, False)
        elif group == 'control':
            # Negation of 't':
            idx = ~self.datasets[target_set]['t']
            tmp = self.subset_by_group(target_set, idx, False)
        elif group == 'all':
            tmp = self.datasets[target_set]
        else:
            raise Exception("Group '{}' not recognized".format(group))
        return tmp


    def subset_by_group(self, target_set, idx, recalculate_r=True):
        """
        Method for creating subset of self.datasets[target_set] where items
        in idx are included.

        Parameters
        ----------
        target_set : str 
            Name of subset to be subsetted (e.g. 'testing_set')
        idx : np.array([bool]) 
            Boolean array for items to be included.
        recalculate_r : bool 
            If true, the revert-label is re-estimated in the subset. 
            Otherwise the previously estimated values are used.

        Returns
        -------
        dict
            A dict with appropriate keys and data.
        """
        X = self.datasets[target_set]['X'][idx, :]
        y = self.datasets[target_set]['y'][idx]
        t = self.datasets[target_set]['t'][idx]
        z = self.datasets[target_set]['z'][idx]
        if recalculate_r:
            r = self._revert_label(y, t)
        else:
            r = self.datasets[target_set]['r'][idx]
        return {'X': X, 'y': y, 't': t, 'z': z, 'r': r}
    
    
    def reduce_size(self, rate):
        """
        Method to reduce the size of the training and validation
        sets while otherwise maintaining their properties. This
        is accomplished simply by random subsampling. This 
        OVERWRITES the training and validation sets in the object!
        Can be used to e.g. test performance of models with different
        training set size and is compatible with the DatsetWrapper-
        class.

        Parameters
        ----------
        rate : float 
            In [0, 1]. The rate at which observations should be
            kept. Observations are randomly sampled without replacement.
        """
        # First training set:
        n_train = len(self['training_set']['y'])
        n_train_to_keep = int(n_train * rate)
        # Create index for random sampling:
        idx = np.random.choice(range(n_train), n_train_to_keep, replace=False)
        self.datasets['training_set'] = self.subset_by_group('training_set', idx)
        
        # Then validation set
        n_val = len(self['validation_set']['y'])
        n_val_to_keep = int(n_val * rate)
        # Create index for random sampling:
        idx = np.random.choice(range(n_val), n_val_to_keep, replace=False)
        self.datasets['validation_set'] = self.subset_by_group('validation_set', idx)


class DatasetWrapper(Dataset):
    """
    Class for wrapping datasets from class above into format accepted
    by torch.utils.data.Dataloader.
    """
    def __init__(self, data):
        """
        Args:
        data (dict): Dictionary with 'X', 'y', 'z', 't', and in
         some cases 'r'.
        """
        self.data = data

    def __len__(self):
        return self.data['X'].shape[0]

    def __getitem__(self, idx):
        # Pytorch needs floats.
        # Perhaps add keys to dict one by one. Current version
        # won't work if not X, y, z, and t are provided.
        X = self.data['X'][idx, :]
        y = self.data['y'][idx].astype(np.float64)
        if 'z' in self.data.keys():
            z = self.data['z'][idx].astype(np.float64)
        if 't' in self.data.keys():
            t = self.data['t'][idx].astype(np.float64)
        if 'r' in self.data.keys():
            r = self.data['r'][idx].astype(np.float64)
            return {'X': X, 'y': y, 'z': z, 't': t, 'r': r}
        else:
            # Datasets don't have 'r' after filtering by group.
            # ('r' would be non-sensical in that case)
            return {'X': X, 'y': y, 'z': z, 't': t}


# Get some data quickly:
def get_criteo_test_data():
    data = DatasetCollection("./datasets/criteo_100k.csv", CRITEO_FORMAT)
    return data

def get_hillstrom_data():
    data = DatasetCollection("./datasets/" + HILLSTROM_FORMAT['file_name'],
                             HILLSTROM_FORMAT)
    return data

def get_voter_data():
    data = DatasetCollection('./datasets/' + VOTER_FORMAT['file_name'],
                             VOTER_FORMAT)
    return data

def get_lenta_test_data():
    data = DatasetCollection("./datasets/lenta_mini.csv", LENTA_FORMAT)
    return data

def get_starbucks_data():
    data = DatasetCollection('./datasets/' + STARBUCKS_FORMAT['file_name'],
                             STARBUCKS_FORMAT)
    return data
