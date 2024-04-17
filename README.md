# Uplift Modeling
Code relating to uplift modeling

## Install package using pip.
```pip install git+https://github.com/trinli/uplift_modeling@packaging_tmp``` for regular installation.
```pip install -e . extras``` for installation supporting gaussian processes and an honest tree implemented in R.

## Run tests for metrics
```python -m uplift_modeling.tests.test_uplift_metrics```


1. Download data to ./datasets/
-http://ailab.criteo.com/criteo-uplift-prediction-dataset/
-https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html
-https://isps.yale.edu/research/data/d001
-https://github.com/joshxinjie/Data_Scientist_Nanodegree/tree/master/starbucks_portfolio_exercise
-https://zenodo.org/record/3653141
2. Extract data to .csv-file.
3. Install python requirements listed in requirements.txt (some might be unnecessary).
4. Create load_data.DatasetCollection object following instructions in load_data.py
5. Pick model from models, train and predict
6. Evaluate performance using uplift_metrics.UpliftMetrics class.

* ./data/ contains files related to data wrangling, with the class load_data.DatasetCollection being at the heart of everything.
* ./experiments/ contains code tying together data wrangling, models, and metrics into experiments run. Some of the results are published.
* ./metrics/ contains code for estimating metrics with the class uplift_metrics.UpliftMetrics estimating the most commonly used metrics in uplift modeling.
* ./models/ contains a wide selection of different uplift models, some published, some unpublished.
* ./slurm/ contains code for running parallel experiments on a cluster using the slurm workload manager
* ./tests/ contains a few tests with the most important one being a test for the metrics package.


# Publications
The following publications are based on the code in this repository:

* "Uplift Modeling with High Class Imbalance" by Otto Nyberg, Tomasz Kuśmierczyk, and Arto Klami. Asian Conference on Machine Learning, PMLR, 2021.
https://proceedings.mlr.press/v157/nyberg21a

* "Exploring Uplift Modeling with High Class Imbalance" by Otto Nyberg and Arto Klami. Data Mining and Knowledge Discovery (accepted for publication), 2023.
https://www.springer.com/journal/10618

* "Quantifying Uncertainty of the Conditional Average Treatment Effect" by Otto Nyberg and Arto Klami. Work-in-progress.


## Running the experiments for "Exploring Uplift Modeling with High Class Imbalance"
1. Store appropriate datasets to ./datasets/ in csv-format
2. Change rate between treated and untreated if desired (e.g. Criteo-uplift 2 was used with 1:1 treated to untreated ratio)
3. For split undersampling experiments, run 'python -m experiments.run_crf_experiment [dataset file] ./datasets/ 1.0 [model] [k_t] [k_c]'
with appropriate parameters, e.g. 
```
python -m experiments.run_crf_experiment starbucks.csv ./datasets/ 1.0 uplift_dc 2 16
```
Here model can be uplift_dc or uplift_rf (the only ones tested compatible with split undersampling), and k_t and k_c can take values larger or equal to 1. 

4. For the other undersampling experiments, run 'python -m experiments.undersampling_experiments [dataset] [undersampling scheme] [model] [calibration method] [output file] [k-values] [positive rate]', e.g. 
```
python -m experiments.undersampling_experiments starbucks naive_undersampling dc_lr isotonic results.csv 8 1
```
This will run a double classifier (a.k.a T-learner) with logistic regression as base learner with naive undersampling with k=8 and tau-isotonic regression for calibration on the starbucks-dataset.


## Running the experiments for "Uplift Modeling with High Class Imbalance"
1. Store appropriate datasets to ./dataset/ in csv-format
2. Run ```python -m uplift_modeling.data.pickle_dataset``` to prepare data appropriately. This normalizes the data and prepares training, validation, and testing sets and creates a new label by running the class-variable transformation. Be patient. The Criteo-uplift 1 dataset is large and we recommend reserving 120GB of RAM for this step. We ran this 10 times to get 10 differently randomized datasets.
3. Run undersampling experiments by running undersampling_experiments.py with suitable parameters, e.g. ```python -m experiments.split_undersampling ./datasets/criteo-uplift.csv123.gz cvt 1,200,10``` (replace '123' with whatever your file is named, 'cvt' refers to class-variable transformation, '1,600,10' indicates "test k from 1 to 600 with a step of 10"). Note that the last print section shows the testing set metrics for the best model.
4. Run isotonic regression experiments, e.g. ```python -m experiments.isotonic_regression_for_calibration ./datasets/criteo-uplift.csv123.gz dclr 3``` (replace '123' with your dataset file, 'dclr' refers to double-classifier with logistic regression, '3' refers to k=3).
5. Results are printed to screen and stored in uplift_results.csv. Look for rows with 'Test description' set to 'testing set'.

The alternative models for both undersampling and isotonic regression experiments are

* 'dc' (or 'dclr'): double-classifier with logistic regression
* 'dcrf': double-classifier with random forest
* 'cvt' (or 'cvtlr'): class-variable transformation with logistic regression
* 'cvtrf': class-variable transformation with random forest

In the paper, we created 10 randomized data sets, ran the code 10 times and averaged the results. For visualizations, use function plot_uplift_curve() in uplift_metrics.py.

# Notes
The code is mostly written by Otto Nyberg as part of my work on my dissertation at the University of Helsinki.
Tomasz Kusmierczyk has contributed to the code and Arto Klami has provided useful feedback.
The file experiments/dirichlet_gp.py requires an exotic environment (old version of python, ancient version of tensorflow etc.). Details for that might be provided later.
