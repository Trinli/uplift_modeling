"""
File for testing CausalML-library.
Just basic tests. Train a regular model, see how long it takes to train,
test it with the CI-estimation using the built-in bootstrap, time that.
"""


import numpy as np
import data.load_data as ld
import metrics.uplift_metrics as um
from random_forestry import RandomForest
from xgboost import XGBRegressor
from causalml.inference.meta import BaseXRegressor


data_format = ld.STARBUCKS_FORMAT  # Change SEED in data format to re-randomize.
data_format['file_name'] = './datasets/' + data_format['file_name']
size = 32000  # Number of observations to include in training set


data = ld.DatasetCollection(data_format['file_name'], data_format=data_format)

# Try base learner
tmp_model = XGBRegressor()  # Regular XGBoost regression model.
tmp_model.fit(data['training_set']['X'], data['training_set']['t'], data['training_set']['y'])

# Try actual uplift model
model = BaseXRegressor(learner=XGBRegressor())
model.fit(data['training_set']['X'], data['training_set']['t'], data['training_set']['y'])
# Training one model without bootstrap on the Starbucks dataset with 32k obesrvations takes two seconds.
pred = model.predict(data['testing_set']['X'])
pred = [item for item in pred[:, 0]]
pred = pred.reshape(-1)  # Drop additional dimension.
tmp_metrics = um.UpliftMetrics(data['testing_set']['y'], pred, data['testing_set']['t'])

print(tmp_metrics)  # Basic use case is working.

# Next try estimating confidence intervals using bootstrap. Also set number of observations per bootstrap (bootstrap_size).
# Default produces 95% confidence intervals (actually so that 2.5% is to the left and 2.5% is to the right of the interval).

ci_model = BaseXRegressor(learner=XGBRegressor())
# fit_predict() is a strange function. Trains the model _and_ predicts confidence intervals only. No uplift.
tmp = ci_model.fit_predict(data['training_set']['X'], data['training_set']['t'], data['training_set']['y'], return_ci=True, n_bootstraps=10)  # Set bootstrap_size?
# Finishes in 10 seconds with n_bootstraps=10.
# tmp[0] is CATE, tmp[1] is lower boundary, tmp[2] is upper boundary.
# fit_predict() does bootstrapping in a way where a model is trained on a subset of the trainind data and the predictions are returned on
# all training observations. Hence, the observations themselves are used for training bootstrap_size/data['training_set']['y'].shape[0] (on average) 
# in their own estimates. There is no function in the library to do this with a separate testing set.
################################
## Maybe create a class that inherits from these but does the bootstrap so that it can be done separately on a testing set?
## It is probably easiest to implement this as an ensemble of BaseXRegressor()-objects.
## It would be equally easy to implement this on top of the BaseXClassifier(), although this has not been presented in the literature.
## Kunzel & al used 10,000 bootstrap iterations! We can probably not do that. Hmm. Default in https://causalml.readthedocs.io/en/latest/_modules/causalml/inference/meta/xlearner.html#BaseXLearner.fit_predict
## is 1,000 iterations (10x less) and 10,000 observations in an iteration. 
## They used 50,000 observations in one iteration.
## Based on their code, Lei & al used X-learner with RF as base-learner (although Künzel also considered a version based on BART).
## This variant of RF CI's are better than the ones provided by Causal Forest. So we can use that.
## Künzel uses a normal approximation as default ("normal approximated CIs"). I.e. predict N times for an observation, estimate mean as uplift,
## and standard deviation. The normal approximated CIs are then mean +- 1.96*sd.
## -Where do I get different predictions for one observation from one model?
## Künzel used "honest RF" and referred to Wager & Athey (2017).
## Honest RF might be implemented here: https://econml.azurewebsites.net/spec/inference.html#subsampled-honest-forest-inference
## These refer to Athey 2019 and talk about confidence intervals using "bootstrap-of-little-bags" (this might be different from Kunzel).
## The sklearn.ensemble.RandomForestRegressor does not implement honest estimation. Maybe use the microsoft package econml.
## Maybe use RandomForestRegressor from sklearn and BootstrapInference from econml to recreate Künzel's model.
## The econml package also contains X-learner.
## econml.grf.RegressionForest seems to be an implementation of the 2017 version, although they say they build on the sklearn version of a tree.
## They also highlight other differences in their implementation.
## -best so far is perhaps the econml bootstrap with sklearn's randomforestregressor? No honest-estimation in sklearn.
## Maybe it is best to run this in R after all. 
################################

## FINALLY GOT random_forestry installed!!!!! I had to update cmake and to install the library with "--no-binary :all:"
## Categorical data needs to be as pandas-dataframe and set to  type "categorical".
## Seems this library does _not_ handle honest estimation as suggested by Athey & Imbens. Instead, they have some sort
## of "OOB-honesty", i.e. out of bootstrap or something.

## Maybe use skgrf for honest forest and BaseXRegressor from causalml or thet X-learner from econml.
## Else, use R-packages.
## causalml seems to require old version of scikit-learn. We can use the X-learner from the econml package instead.
## skgrf seems to require scikit-learn version < 1.0. Latest before 1.0 is 0.24.2. 
## Also, my env3 is completely messed up right now. Numba-optimization is probably not working. Run tests?

## The only "working" library is grf for R. I don't know what parameters are needed to reproduce Honest RF
## and I don't know if there is a library for R that will reproduce the X-learner.
## bartMachine, which could potentially run the BART-model from 2011 (Hill), would require Java and I am not going
## to install Java on my computer.
## I have installed the newest version of Cython using brew, although this one was not added to PATH.
## I also installed an older version of Cython using pip in env_test.
## I also tried other versions of Python using pyvenv. 
## I am thinking I also changed some other version of c-compilers. Not sure which one, though.
## econml.grf also has random forest implementations. One of them might reproduce the honest random forest by Athey and Imbens
## with suitable parameters.

## skgrf
## A library for python providing the same functions as grf for R. GRF is a generalized version of the honest random forest,
## hence it could be possible to implement the original honest random forest with suitable parameters.
## skgrf requires version <1.0 of scikit-learn. The latest version of scikit-learn before 1.0 is 0.24.2.
## I am unable to install scikit-learn 0.24.2. One issue might be that cython 3.0.8 (current version) requires that
## the code is python3, whereas the old version of scikit-learn might be written in python2. I tried installing
## an older version of cython, but installation still failed at compilation.
## -I have no idea how to solve this.

## econml
## This might currently be the most promising library.

## causalml
## This is now installed in env_test. It seems this could be promising, if I was able to figure out which version
## of the random forests implements the honest random forest. 

# Estimate average width of credible intervals (assuming 95% CI's)
widths = [item1 - item2 for item1, item2 in zip(tmp[1], tmp[2])]
average_width = np.mean(widths)
print(average_width)

# Would like to figure out how often the true CATE is within the predicted interval. Cannot do this on empirical data.

# All 

tmp_2 = ci_model.predict(data['testing_set']['X'])

### Let's try random forestry
model = RandomForest()
model.fit(data['training_set']['X'], data['training_set']['r'])
pred = model.predict(data['testing_set']['X'])

