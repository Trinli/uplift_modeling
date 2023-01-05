"""
This is a test for comparing the stability of AUUC and the Qini-coefficient,
two metrics commonly used in uplift modeling. Diemert & al. (2019?) claimed
that the Qini-coefficient is better than AUUC, whereas I claim the opposite.
The Qini-coefficient lacks a sound theoretical basis, although in the balanced
case where N_t = N_c, it is an approximation of AUUC.

The big problem with the Qini-coefficient is that it favors a model that
separates treatment samples from control samples. While this should not be
possible, it will occur due to random fluctuation and make the qini-coefficient
more unstable.

Notes:
Test on Criteo data with 1k iterations, 2M samples in testing set, imbalancing for
both treatment class and control class separately. Dropping 50% of the highest 
50% scoring samples in either treatment or control group hence biasing the data.
AUUC: 
0.0003691706929532018, std: 5.845879202387649e-05, std[relative]: 0.15835165992249298
Qini: 
0.23338000588790114, std: 0.037993377312989865, std[relative]: 0.16279619656552383
Correlation between auuc and qini values
(0.7282808231659305, 5.434373802974609e-166)
Comparison where control samples were dropped
AUUC imbalanced: 
0.0003636627550300545, std: 5.905715412773949e-05, std[relative]: 0.16239538778959858
Qini imbalanced: 
0.22608598436349758, std: 0.038834828020364705, std[relative]: 0.17177017022836172
Correlation between imbalanced auuc and qini values
(0.7135542421134905, 2.1262131430458446e-156)
Comparison where treatment samples were dropped
AUUC imbalanced: 
0.0003338869406330768, std: 6.39074363380323e-05, std[relative]: 0.1914044203611516
Qini imbalanced: 
0.18474149486627084, std: 0.04172724658478379, std[relative]: 0.22586829566897773
Correlation between imbalanced auuc and qini values
(0.6527281046782553, 1.8348661026510644e-122)

In principle, the following should all be 1 if we wanted the metrics to fix the
bias in data:
auuc_c/auuc_0 = 95.51%
auuc_t/auuc_0 = 90.44%
qini_c/qini_0 = 96.87%
qini_t/qini_0 = 79.16%
-Clearly the auuc deteriorates less.
Treatment dropped: 
std 18% higher for Qini-coefficient.
auuc dropped by 9.96&, std increased by 20.87%
qini dropped by 20.84%, std increased by 38.74%
-The qini metrics are "twice as unstable" as the AUUC measured here.
Control dropped:
AUUC dropped by 1.49%, std increased by 2.56%
Qini dropped by 3.12%, std increased by 5.59%
The ratio of auuc/qini is in the balanced case 0.00158, in the case where control
were dropped 0.00161, and in the case where treatment samples were dropped 0.00181.
The ration of auuc_0/qini_x is 0.00158 in the balanced case, 0.00163 when qini_x
refers to qini_c, and 0.00200 when qini_x is qini_t (treatment dropped).

As we did not drop all samples, but simply some of the higher scoring treatment
and control samples (in separate tests), the metrics should remain constant if they
are to be resilient against this kind of bias. We see that the standard deviation
for the qini coefficient is approximately twice that of the AUUC

Note how large the relative standard deviation is! Even with 2M samples in the testing
set, we end up with values that vary >15% 32% of the time for both metrics!
Also note that the relative improvement is relative to the metric itself (e.g. not
E_r(y|random) as in our previous tests).
I would have expected higher correlation. Correlation of .7 still implies that half
of the variance is unexplained. If these metrics were truly to measure the same thing
I would expect the correlation to exceed .9.
"""
import numpy as np
import gzip
import pickle
from scipy.stats.stats import pearsonr
import models.double_classifier as double_classifier
import data.load_data as load_data
import metrics.uplift_metrics as uplift_metrics


N_ITERATIONS = 1000
SAMPLE_SIZE = 1000000

# 1. Load data
#data = load_data.get_hillstrom_data()
#data = load_data.DatasetCollection('./datasets/criteo-uplift.csv1740898770.pickle.gz',
#                                   load_data.DATASET_FORMAT)
path = './datasets/criteo-uplift.csv1740898770.pickle.gz'
fp = gzip.open(path, 'rb') if path.endswith(".gz") else open(path, 'rb')
data = pickle.load(fp)
fp.close()
# Add something for Criteo-data here. That is more interesting

# 2. Train e.g. DC-LR on 1/10th of the data
model = double_classifier.LogisticRegression()
model.fit(data)

# 3. Draw 100 random subsamples from the test data and estimate metrics
# for both Qini and AUUC. Store values.

def drop_samples(y, pred, t, fraction=.5, group=False):
    """
    Function for dropping low scoring control samples and high scoreing
    treatmet samples.

    Args:
    group (bool): True for treatment, False for control
    """
    # 1. Identify low scoring samples (10%?)
    idx = np.argsort(pred)[::-1]  # Decreasing order
    y_tmp = y[idx]
    pred_tmp = pred[idx]
    t_tmp = t[idx]
    # Pick fraction of highest scoring control samples and drop
    drop_idx = np.array([j for j, group_tmp in enumerate(t) if group_tmp == group])
    # Number of samples in selected group:
    n_group = sum(t_tmp == group)
    drop_idx = drop_idx[0:int(fraction * n_group)]
    # Randomly choose a subsample of these to drop:
    drop_idx = np.random.choice(drop_idx, int(len(drop_idx)*.5))
    y_tmp = np.delete(y_tmp, drop_idx)
    pred_tmp = np.delete(pred_tmp, drop_idx)
    t_tmp = np.delete(t_tmp, drop_idx)
    return (y_tmp, pred_tmp, t_tmp)

auuc = []
qini = []
auuc_t_imbalanced = []
qini_t_imbalanced = []
auuc_imbalanced = []
qini_imbalanced = []
dataset_size = data['testing_set']['X'].shape[0]
for i in range(N_ITERATIONS):
    # In the Criteo data, the testing set already contain 7 M samples
    sampling_idx = np.random.randint(dataset_size, size=SAMPLE_SIZE)
    tmp_X = data['testing_set']['X'][sampling_idx, :]
    tmp_y = data['testing_set']['y'][sampling_idx]
    tmp_t = data['testing_set']['t'][sampling_idx]
    predictions = model.predict_uplift(tmp_X)
    metrics = uplift_metrics.UpliftMetrics(tmp_y, predictions, tmp_t)
    auuc.append(metrics.auuc)
    qini.append(metrics.qini_coefficient)
    # Treatment samples
    y_tmp_t, pred_tmp_t, t_tmp_t = drop_samples(tmp_y, predictions, tmp_t, group=True)
    imbalanced_t_metrics = uplift_metrics.UpliftMetrics(y_tmp_t, pred_tmp_t, t_tmp_t)
    auuc_t_imbalanced.append(imbalanced_t_metrics.auuc)
    qini_t_imbalanced.append(imbalanced_t_metrics.qini_coefficient)
    # Control samples:
    y_tmp, pred_tmp, t_tmp = drop_samples(tmp_y, predictions, tmp_t, group=False)
    imbalanced_metrics = uplift_metrics.UpliftMetrics(y_tmp, pred_tmp, t_tmp)
    auuc_imbalanced.append(imbalanced_metrics.auuc)
    qini_imbalanced.append(imbalanced_metrics.qini_coefficient)


# 4. Estimate mean and variance for both metrics
# Note that the variance should also be reported as % of mean.
# Perhaps also estimate correlation between the twoe
auuc_mean = np.mean(auuc)
auuc_std = np.std(auuc)
auuc_relative_std = auuc_std / auuc_mean
qini_mean = np.mean(qini)
qini_std = np.std(qini)
qini_relative_std = qini_std / qini_mean
correlation = pearsonr(auuc, qini)

print("AUUC: {}, std: {}, std[relative]: {}".format(auuc_mean,
                                             auuc_std,
                                             auuc_relative_std))
print("Qini: {}, std: {}, std[relative]: {}".format(qini_mean,
                                             qini_std,
                                             qini_relative_std))
print("Correlation between auuc and qini values")
print(correlation)

# Same for the imbalanced case:
auuc_mean_i = np.mean(auuc_imbalanced)
auuc_std_i = np.std(auuc_imbalanced)
auuc_relative_std_i = auuc_std_i / auuc_mean_i
qini_mean_i = np.mean(qini_imbalanced)
qini_std_i = np.std(qini_imbalanced)
qini_relative_std_i = qini_std_i / qini_mean_i
correlation_i = pearsonr(auuc_imbalanced, qini_imbalanced)

print("Comparison where control samples were dropped")
print("AUUC imbalanced: {}, std: {}, std[relative]: {}".format(auuc_mean_i,
                                                        auuc_std_i,
                                                        auuc_relative_std_i))
print("Qini imbalanced: {}, std: {}, std[relative]: {}".format(qini_mean_i,
                                                        qini_std_i,
                                                        qini_relative_std_i))
print("Correlation between imbalanced auuc and qini values")
print(correlation_i)


# Treatment case
# Same for the imbalanced case:
auuc_mean_i = np.mean(auuc_t_imbalanced)
auuc_std_i = np.std(auuc_t_imbalanced)
auuc_relative_std_i = auuc_std_i / auuc_mean_i
qini_mean_i = np.mean(qini_t_imbalanced)
qini_std_i = np.std(qini_t_imbalanced)
qini_relative_std_i = qini_std_i / qini_mean_i
correlation_i = pearsonr(auuc_t_imbalanced, qini_t_imbalanced)

print("Comparison where treatment samples were dropped")
print("AUUC imbalanced: {}, std: {}, std[relative]: {}".format(auuc_mean_i,
                                                        auuc_std_i,
                                                        auuc_relative_std_i))
print("Qini imbalanced: {}, std: {}, std[relative]: {}".format(qini_mean_i,
                                                        qini_std_i,
                                                        qini_relative_std_i))
print("Correlation between imbalanced auuc and qini values")
print(correlation_i)
