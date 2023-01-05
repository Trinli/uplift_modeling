"""
Classes for handling datasets for uplift experiments.

Notes:
Support for ordinal variables could perhaps be added. Somewhat involved
 as labels needs to be mapped to values (e.g. likert-scale).
Perhaps add tests for load_data.
Support for other dependent variables than simply 0/1 could be added
 following the same logic used with treatment labels (t_labels).
Perhaps expand the documentation for the data format.
Not tested for h5py-files.
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
DATA_FORMAT = CRITEO_FORMAT  # Rename the above without breaking stuff.

# This format is for the Hillstrom Mine that data -challenge dataset
HILLSTROM_FORMAT_1 = {'file_name':
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
    Class for handling dataset related operations (loading, normalization,
     dummy-coding, access, group-subsetting, undersampling).

    Methods:
    __init__(file_name (str), mode={None, 'test'}):
     Initialization loads data from specified csv-file.
    _normalize_data(): Normalizes data. Default normalization is 'v3'
     specified in DATA_NORMALIZATION and includes centering and setting
     of variance to 1 for features.
    _load_data(): Imports csv-files
    _extract_data(): Reads in features, class-labels, and treatment-label
     following the data specified in self.data_format.
    _create_subsets(): Creates training, validation, and testing sets with
     splits 1/2, 1/4, 1/4, and further adds a second training, and two
     validation sets with 6/16, 3/16, 3/16 splits. This leaves the testing
     set untouched.
    __add_set(): Auxiliary function for _create_subsets().
    __getitem__(): Method for accessing all data once loaded in the initialization.
    _undersample(): Method for undersampling data both to 1:1 and 1:1:1:1.
    __subset_by_group(): Method for selecting only treatment, control, or all data.
    """
    def __init__(self, file_name,
                 data_format=DATA_FORMAT):
        """
        Method for initializing object of class DataSet that will handle
        all dataset related issues.

        Attributes:
        file_name (str): {'criteo-uplift.h5', 'criteo100k.csv', 'criteo-uplift.csv'}
        nro_features (int): number of features in dataset. Assumed to be in columns 0:nro_features.
        seed (int): random seed to use for split of data.

        Notes:
        Potentially dataloaders could also be stored in this object i a dict
         in the same way datasets are stored now. Would require carrying information
         on treatment/control.
        """
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
        Method for loading data from file. Currently supports csv-files.

        Maybe store the data into features (X), y, and t at this point already?
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
                # This becomes misleading as we reformat the data:
                # self.col_names = tmp_data[1]
                # Drop title row:
                tmp_data = tmp_data[1:]
            # Perhaps make class and treatment labels binary?
            # Everything is one array which prevents this.
        else:
            raise Exception("The file needs to be a .csv-file.")

        # Parse data into suitable format following format specified in
        # self.data_format:
        tmp_X = None
        for col, _ in enumerate(tmp_data[0]):
            # Parse into suitable format
            # Either it is continuous or categorical.
            # Some columns are also treatment-group labels etc ?!?
            # if column is feature:
            if col in self.data_format['X_idx']:
                if col in self.data_format['continuous_idx']:
                    tmp = np.array([[row[col]] for row in tmp_data])
                    tmp = tmp.astype(self.data_format['data_type'])
                    tmp = self._normalize_data(tmp)
                elif col in self.data_format['categorical_idx']:
                    tmp_col = [row[col] for row in tmp_data]
                    keys = np.unique(tmp_col)
                    tmp = np.zeros([len(tmp_col), len(keys)], dtype=
                                   self.data_format['data_type'])
                    for i, key in enumerate(keys):
                        for j, _ in enumerate(tmp_col):
                            if tmp_col[j] == key:
                                tmp[j, i] = 1
                # Add new features to tmp_array
                if tmp_X is None:
                    tmp_X = tmp
                else:
                    tmp_X = np.concatenate([tmp_X, tmp], axis = 1)
            elif col == self.data_format['t_idx']:
                # Binary group label
                # True indicates treatment group
                tmp = [row[col] for row in tmp_data]
                tmp_t = np.array([item == self.data_format['t_labels'][0] for
                                  item in tmp])
            elif col == self.data_format['y_idx']:
                if 'y_labels' in self.data_format.keys():
                    # I.e. if labels are given for the class:
                    tmp = [row[col] for row in tmp_data]
                    tmp_y = np.array([item == self.data_format['y_labels'][0] for
                                      item in tmp])
                else:
                    # No 'y_idx' is specified, reading in as boolean.
                    tmp_y = np.array([row[col] for row in tmp_data])
                    tmp_y = tmp_y.astype(np.bool_)
                    # tmp_y = tmp_y.astype(self.data_format['data_type'])
        # Class-variable transformation:
        tmp_z = tmp_y == tmp_t

        # Keep only samples that belong to specified treatment groups:
        tmp = [row[self.data_format['t_idx']] for row in tmp_data]
        group_idx = [item in self.data_format['t_labels'] for item in
                     tmp]
        group_idx = np.array(group_idx)
        self.X = tmp_X[group_idx, :]
        self.y = tmp_y[group_idx]
        self.t = tmp_t[group_idx]
        self.z = tmp_z[group_idx]

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
        print("Estimated effect size for treatment: {:.2f}%".format(effect_size * 100))


    def _shuffle_data(self):
        if self.data_format['random_seed'] is not None:
            print("Random seed set to {}.".format(self.data_format['random_seed']))
            # Set random seed to get same split for all experiments
            np.random.seed(self.data_format['random_seed'])
        shuffling_idx = np.random.choice([item for item in range(len(self.y))],
                                         len(self.y), replace=False)
        self.X = self.X[shuffling_idx, :]
        self.y = self.y[shuffling_idx]
        self.t = self.t[shuffling_idx]
        self.z = self.z[shuffling_idx]
        # self.r does not exist at this point yet.
        #  self.r = self.r[shuffling_idx]
        # Reset seed:
        np.random.seed(None)  # Uses time

    def _create_subsets(self):
        """
        Method for creating training, validation, and testing sets plus
        an additional training_set_2, validation_set_2a, and validation_set_2b
        to be used for deciding on early stopping when training neural networks.
        Training, validation, and testing sets are split 50:25:25, and
        training2, validation_set_2a, validation_set_2b, and testing set are
        split 6/16:3/16:3/16:4/16. Note that in both cases, the testing sets are
        identical in all aspects.
        """
        # Using a 50/25/25 split
        n_samples = self.X.shape[0]
        # Add usable datasets such as training set to self.datasets:
        self.__add_set('training_set', 0, n_samples // 2)
        self.__add_set('validation_set', n_samples // 2, n_samples * 3 // 4)
        self.__add_set('testing_set', n_samples * 3 // 4, n_samples)
        # Add also slightly different split that enables early stopping of
        # neural networks using separate validation set:
        self.__add_set('training_set_2', 0, n_samples * 6 // 16)
        self.__add_set('validation_set_2a', n_samples * 6 // 16, n_samples * 9 // 16)
        self.__add_set('validation_set_2b', n_samples * 9 // 16, n_samples * 12 // 16)
        # Undersampled training set:
        # Not creating the 1:1:1:1 sets by default anymore as of 12.3.2021
        tmp = self._undersample('training_set', '1:1')
        self.datasets.update({'undersampled_training_set_11': tmp})
        #tmp = self._undersample('training_set', '1:1:1:1')
        #self.datasets.update({'undersampled_training_set_1111': tmp})
        tmp = self._undersample('training_set_2', '1:1')
        self.datasets.update({'undersampled_training_set_2_11': tmp})
        #tmp = self._undersample('training_set_2', '1:1:1:1')
        #self.datasets.update({'undersampled_training_set_2_1111': tmp})

    def _revert_label(self, y_vec, t_vec):
        """
        Class-variable transformation ("revert-label") approach
        following Athey & Imbens (2015).

        Args:
        y_vec (np.array([float])): Array of conversion values in samples.
        t_vec (np.array([bool])): Array of treatment labels for same samples.
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

    def __add_set(self, name, start_idx, stop_idx):
        """
        Auxiliary function for _create_subsets(). Adds usable datasets as dicts
        with X, y, t, and z. 'z' here refers to class-variable transformation
        following Jaskowski & Jaroszewicz (2012).
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

    def add_set(self, name, start_idx, stop_idx):
        """
        Public method of above.
        """
        self.__add_set(name, start_idx, stop_idx)

    def _normalize_data(self, vector):
        """
        Method for normalizing data.

        Attributes:
        version (str): {'v1', 'v2', 'v3'}
         -v1 and v2 are for testing purposes
         -v3 (default) centers data and sets variance to unit (1)
        """
        # This normalizes features as stated in Diemert & al.
        if self.data_format['normalization'] is None:
            pass
        elif self.data_format['normalization'] == 'v1':
            tmp = normalize(vector)
        elif self.data_format['normalization'] == 'v2':
            # Set variance to 1
            tmp = vector / np.std(vector)
        elif self.data_format['normalization'] == 'v3':
            # Set mean to 0 and variance to 1:
            tmp = vector - np.mean(vector)
            tmp = tmp / np.std(tmp)
        return tmp

    def _undersample(self, name, method):
        """
        Method that adds undersampled data as np.array into dataset dict
        together with 'X', 'y', and 't'.

        Args:
        name (str): Name of set to be added to
        # Where do we get the info on which set to undersample? Always
        # training_set?
        1:1 undersampling only for SVM double classifier (with separated
        (treatment and control sets)
        1:1:1:1 used together with CVT (any base learner)

        Notes:
        Only the '1:1' is theoretically sound in this method. Changing to 1:1:1:1
        is just a quick hack. k_undersampling() is better.
        """
        tmp = self.datasets[name]
        tot_samples = len(tmp['t'])
        if method == '1:1':
            # 1:1, treatment:control ratio
            # Smaller of these is the number of samples we want
            n_samples = int(np.min([sum(tmp['t']), sum(~tmp['t'])]))
            treatment_idx = np.array([i for i, t in zip(range(tot_samples), tmp['t'])
                                      if t])
            control_idx = np.array([i for i, t in zip(range(tot_samples), tmp['t'])
                                    if not t])
            # Shuffle:  -no random seed?
            np.random.shuffle(treatment_idx)
            np.random.shuffle(control_idx)
            idx = np.concatenate([treatment_idx[:n_samples],
                                  control_idx[:n_samples]])
        elif method == '1:1:1:1':
            print("Note that the 1:1:1:1 method is not theoretically sound. Perhaps deprecate?")
            # positive_treatment:negative_treatment:positive_control:
            # negative_control, 1:1:1:1
            # Looking for min of positive or negative classes in any group:
            n_samples = int(np.min([np.sum(tmp['t'] * tmp['y']),
                                       np.sum(~tmp['t'] * tmp['y']),
                                       np.sum(tmp['t'] * (tmp['y'] == 0)),
                                       np.sum(~tmp['t'] * (tmp['y'] == 0))]))
            pos_treatment_idx = np.array([i for i, t, y in zip(range(tot_samples),
                                                               tmp['t'], tmp['y'])
                                          if t & (y == 1)])
            neg_treatment_idx = np.array([i for i, t, y in zip(range(tot_samples),
                                                               tmp['t'], tmp['y'])
                                          if t & (y == 0)])
            pos_control_idx = np.array([i for i, t, y in zip(range(tot_samples),
                                                             tmp['t'], tmp['y'])
                                        if (not t) & (y == 1)])
            neg_control_idx = np.array([i for i, t, y in zip(range(tot_samples),
                                                             tmp['t'], tmp['y'])
                                        if (not t) & (y == 0)])
            # Shuffle so that [:n_samples] is a random sample:
            np.random.shuffle(pos_treatment_idx)
            np.random.shuffle(neg_treatment_idx)
            np.random.shuffle(pos_control_idx)
            np.random.shuffle(neg_control_idx)
            # Take #n_samples from each type and concatenate:
            idx = np.concatenate([pos_treatment_idx[:n_samples],
                                  neg_treatment_idx[:n_samples],
                                  pos_control_idx[:n_samples],
                                  neg_control_idx[:n_samples]])
        else:
            raise Exception("The defined undersampling method, " +
                            "{}, does not exist".format(method))
        # Shuffle index for good measure (prevent any idiosyncrasies in
        # algorithms to have weird effects)
        np.random.shuffle(idx)
        X = tmp['X'][idx, :]
        y = tmp['y'][idx]
        t = tmp['t'][idx]
        z = tmp['z'][idx]
        # Revert-label does not make sense together with undersampling.
        # (techically it is possible to calculate, but the normalization is
        # precisely there to make undersampling unnecessary.)
        # At minimum, it would need to be recalculated for a subset.
        return {'X': X, 'y': y, 't': t, 'z': z}


    def k_undersampling(self, k, group_sampling='11'):
        """
        Method returns a training set where the rate of positive samples
        is changed by a factor of k by either reducing the number of
        negative samples or increasing the number of positive samples.
        The method also changes the sampling rate of treatment vs. control
        samples to 1:1.
        This is suitable for class-variable transformation.

        Args:
        k (int): If None, a balanced k is deduced from the data. Otherwise
         this number will determine the change in positive rate in the data.
         group_sampling (str): 'natural' implies no change in group sampling
         rate, i.e. the number of samples in the treatment and control groups
         stay constant. 
         '11' indicates that there should be equally many treatment and
         control samples. This is useful with CVT and enforces 
         p(t=0) = p(t=1)).

        Notes:
        If k is very large the number of negative samples might drop to zero,
        or conversely if k is very small the number of positive samples might
        drop to zero. There is not a check for this implemented. The implementation
        ensures at least one negative sample is retained.
        -No, it does not. Negative samples can drop to zero.
        """
        # Number of positives in treatment group:
        t_data = self['training_set', None, 'treatment']
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

        c_data = self['training_set', None, 'control']
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
        np.random.shuffle(idx)
        tmp_X = self['training_set']['X'][idx, :]
        tmp_y = self['training_set']['y'][idx]
        tmp_t = self['training_set']['t'][idx]
        tmp_z = self['training_set']['z'][idx]
        # We will also need 'r' here now (MSE-gradient)!
        tmp_r = self._revert_label(tmp_y, tmp_t)

        return {'X': tmp_X, 'y': tmp_y, 'z': tmp_z, 't': tmp_t, 'r': tmp_r}

    def undersampling(self, k_t, k_c=None, group_sampling=None,
                      seed=None, target_set='training_set'):
        """
        Method to undersample the training set. Method returns a dictionary
        containing labels 'y', features 'X', treatment group 't',
        the class-variable transformation 'z', and the revert-label
        transformation 'r'. The undersampling is performed so that
        p(y=1) in the original data equals p(y=1) / k_t (or k_c) in
        the undersampled data for treated and control samples separately.

        Args:
        k_t (float): Value of k for undersampling of treated samples.
         Value in [0, inf]. For values below 1, positive samples are
         dropped.
        k_c (float): Gets value k_t unless k_c is specified. I that case
         it works like k_undersampling.
        group_sampling (str): If set to '11' there will be equally many
         treatment and control samples.
        seed (int): Random seed for numpy. If set to None, random seed is
         not set.
        target_set (str): Which set to sample the undersampled dataset from. Needs
         to exist as key in DatasetCollection.
        """
        if k_c is None:
            # Use same k for both treated and control samples.
            k_c = k_t

        # Number of positives in treatment group:
        t_data = self[target_set, None, 'treatment']
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

        c_data = self[target_set, None, 'control']
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
        Naive version of undersampling where negative (majority class) samples
        are dropped from both treated and control samples with equal probability.
        Using 'k' here instead of e.g. 'p' as undersampling factor to keep
        similarity to other undersampling methods.
        
        Args:
        k (float): Undersampling factor. In ]0, inf], although upper boundary comes
        from number of samples that can be dropped before dropping _all_ majority
        class samples. k > 1 will lead to negative samples being dropped and k < 1
        to positive ones being dropped so that p(y=1) = k * \tilde{p}(y=1).
        """
        # 1. Just calculate what k_t and k_c this naive k corresponds to
        # and call undersampling() with appropriate parameters.
        n_samples = len(self['training_set', None, 'all']['y'])
        n_positives = sum(self['training_set', None, 'all']['y'])
        n_negatives = n_samples - n_positives
        conversion_rate = n_positives / n_samples
        assert k > 0, "k needs to be larger than 0."
        assert k * conversion_rate <= 1, "Given k too large for data."
        # new_conversion_rate = k * conversion_rate
        if k >= 1:
            # k > 1 implies negative samples are dropped.
            # Number of negative samples to be dropped:
            # (again assuming we are dropping negative samples)
            n_neg_drop = n_samples * (1 - 1 / k)  # This when k >= 1.
            drop_rate = n_neg_drop / n_negatives
            # Treated samples:
            tmp_pos_n = sum(self['training_set', None, 'treatment']['y'])
            tmp_n = len(self['training_set', None, 'treatment']['y'])
            tmp_samples_to_drop = (tmp_n - tmp_pos_n) * drop_rate
            k_t = (tmp_pos_n / (tmp_n - tmp_samples_to_drop)) / (tmp_pos_n / tmp_n)
            # Control samples:
            tmp_pos_n = sum(self['training_set', None, 'control']['y'])
            tmp_n = len(self['training_set', None, 'control']['y'])
            tmp_samples_to_drop = (tmp_n - tmp_pos_n) * drop_rate
            k_c = (tmp_pos_n / (tmp_n - tmp_samples_to_drop)) / (tmp_pos_n / tmp_n)
        elif 0 < k < 1:
            #n_pos_drop = n_samples / k - n_negatives # Doesn't apply for k < 1.
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
            tmp_pos_n = sum(self['training_set', None, 'treatment']['y'])
            tmp_n = len(self['training_set', None, 'treatment']['y'])
            # tmp_neg_n = tmp_n - tmp_pos_n
            tmp_samples_to_drop = tmp_pos_n * drop_rate
            k_t = (tmp_pos_n - tmp_samples_to_drop) /\
                (tmp_n - tmp_samples_to_drop) /\
                    (tmp_pos_n / tmp_n)
            # Control samples:
            tmp_pos_n = sum(self['training_set', None, 'control']['y'])
            tmp_n = len(self['training_set', None, 'control']['y'])
            # tmp_neg_n = tmp_n - tmp_pos_n
            tmp_samples_to_drop = tmp_pos_n * drop_rate
            k_c = (tmp_pos_n - tmp_samples_to_drop) /\
                (tmp_n - tmp_samples_to_drop) /\
                    (tmp_pos_n / tmp_n)
            # k_c = (tmp_pos_n - tmp_samples_to_drop) /\
            #    (tmp_pos_n - tmp_samples_to_drop + tmp_neg_n)
        else:
            # Should never end up here.
            raise ValueError("k not valid.")
        return self.undersampling(k_t=k_t, k_c=k_c)

    def __getitem__(self, *args):
        """
        Shorthand method to access self.datasets' contents. This function will
        return either a numpy-array or a pytorch dataloader depending on parameters.

        Args:
        args[0] = name (str): name of key in data.datasets to access, e.g.
         'training_set'.
        args[1] = undersampling {None, '11', '1111'}: None causes no undersampling
         at all, '11' results in treatment and control groups being equally large,
         '1111' results in '11' and #positive and #negative in both groups to be
         equally large.
        args[2] = group {'all', 'treatment', 'control'}: 'all' and None both
         return all data. 'treatment' returns samples that were treated etc.

        Notes:
        *No solution for fetching data with CVT by Athey & Imbens (2015)
        """
        # Handle input arguments:
        group = 'all'  # Subset for treatment or control?
        if isinstance(args[0], str):  # Only name of dataset passed.
            name = args[0]
        elif isinstance(args[0], tuple):  # Multiple arguments were passed
            name = args[0][0]
            if len(args[0]) > 1:
                undersampling = args[0][1]
                if (undersampling is not None) and name == 'training_set':
                    name = 'undersampled_' + name + '_' + str(undersampling)
                elif undersampling is not None:
                    print("Currently no undersampled datasets for other " +
                          "than training set.")
                if len(args[0]) > 2:
                    group = args[0][2]
                    if len(args[0]) > 3:
                        raise Exception("Too many arguments.")
        else:
            raise Exception("Error in __getitem__()")

        # Store approproate data in tmp:
        if group == 'treatment':
            idx = self.datasets[name]['t']
            tmp = self.__subset_by_group(name, idx, False)
        elif group == 'control':
            # Negation of 't':
            idx = ~self.datasets[name]['t']
            tmp = self.__subset_by_group(name, idx, False)
        elif group == 'all':
            tmp = self.datasets[name]
        else:
            raise Exception("Group '{}' not recognized".format(group))
        return tmp

    def __subset_by_group(self, name, idx, recalculate_r=True):
        """
        Method for creating subset of self.datasets[name] where items
        in idx are included.

        name (str): Name of subset to be subsetted (e.g. 'testing_set')
        idx (np.array([bool])): Boolean array for items to be included.
        recalculate_r (bool): If true, the revert-label is re-estimated
         in the subset. Otherwise the previously estimated values are used.

        Notes:
        -This method creates new arrays.
        -Storing these would double the need for memory.
        """
        X = self.datasets[name]['X'][idx, :]
        y = self.datasets[name]['y'][idx]
        t = self.datasets[name]['t'][idx]
        z = self.datasets[name]['z'][idx]
        if recalculate_r:
            r = self._revert_label(y, t)
        else:
            r = self.datasets[name]['r'][idx]
        return {'X': X, 'y': y, 't': t, 'z': z, 'r': r}
    
    def subset_by_group(self, name, idx):
        """
        Public version of above method.
        """
        return self.__subset_by_group(name, idx)
    
    def subsample(self, rate):
        """
        Method to subsample training and validation set. This 
        OVERWRITES the training and validation sets in the object!
        Can be used to e.g. test performance of models with different
        training set size.
        'z' (class-variable transformation) stays as is. 'r' is re-estimated
        with current data.
        Normalization stays as is.
        
        Args:
        rate (float): In [0, 1]. The rate at which observations should
         be kept. Observations are randomly sampled without replacement.

        Notes:
        Does not touch 'training_set_2' nor related validation sets.
        -Are we using these? These sets were added for neural net cross-
         validation.
        """
        # First training set:
        n_train = len(self['training_set']['y'])
        n_train_to_keep = int(n_train * rate)
        # Create index for random sampling:
        idx = np.random.choice(range(n_train), n_train_to_keep, replace=False)
        self.datasets['training_set'] = self.__subset_by_group('training_set', idx)
        
        # Then validation set
        n_val = len(self['validation_set']['y'])
        n_val_to_keep = int(n_val * rate)
        # Create index for random sampling:
        idx = np.random.choice(range(n_val), n_val_to_keep, replace=False)
        self.datasets['validation_set'] = self.__subset_by_group('validation_set', idx)


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
    data = DatasetCollection("./datasets/criteo_100k.csv", DATA_FORMAT)
    return data

def get_hillstrom_data():
    data = DatasetCollection("./datasets/" + HILLSTROM_FORMAT_1['file_name'],
                             HILLSTROM_FORMAT_1)
    return data

def get_voter_data():
    data = DatasetCollection('./datasets/' + VOTER_FORMAT['file_name'],
                             VOTER_FORMAT)
    return data

def get_lenta_test_data():
    data = DatasetCollection("./datasets/lenta_mini.csv", LENTA_FORMAT)
    return data
