# Uplift Modeling
Code relating to uplift modeling

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


## Publications
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


## Notes
The code is mostly written by Otto Nyberg as part of my work on my dissertation at the University of Helsinki.
Tomasz Kusmierczyk has contributed to the code and Arto Klami has provided useful feedback.
The file experiments/dirichlet_gp.py requires an exotic environment (old version of python, ancient version of tensorflow etc.). Details for that might be provided later.
