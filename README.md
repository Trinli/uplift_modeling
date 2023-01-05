# uplift
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

## Notes
The code is mostly written by Otto Nyberg as part of my work on my dissertation at the University of Helsinki.
Tomasz Kusmierczyk has contributed to the code and Arto Klami has provided useful feedback.
The file experiments/dirichlet_gp.py requires an exotic environment (old version of python, ancient version of tensorflow etc.). Details for that might be provided later.
