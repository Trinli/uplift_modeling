"""
Format-dictionaries a few datasets as used by uplift_modeling.load_data.UpliftDataset.

-Check whether the LENTA_FORMAT is correct. It probably is not.
"""

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
