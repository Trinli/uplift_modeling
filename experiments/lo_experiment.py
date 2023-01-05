"""
Simple program to test how well the model proposed by Lo (2002)
works on Criteo-uplift 1.

The results were perfectly in line with the ones of DC-LR. This is
not surprising at all as this is essentially a data-sharing version
of DC-LR, and as we have lots of data, the data sharing brings no
additional benefit.
AUUC (Criteo-uplift 1, full data): 0.0003649022629000618 (0.17638044816744103 over random)
AUUC (Criteo1-1to1.csv): 0.0004029555108638252 (0.19356075090738342 over random)
AUUC of DC-LR on Criteo1_1to1.csv is 0.000403. 
AUUC of DC-LR on full Criteo-uplift 1 is 0.000362.

-Should the "data-sharing" really improve the results?
"""
from data import load_data
from models import s_learner
from metrics import uplift_metrics


def main(data):
    model = s_learner.LoUplift()
    model.fit(data['training_set']['X'], data['training_set']['y'], data['training_set']['t'])
    predictions = model.predict_uplift(data['testing_set']['X'])
    metrics = uplift_metrics.UpliftMetrics(data['testing_set']['y'], predictions, data['testing_set']['t'])
    print(metrics)
    metrics.write_to_csv('./results/lo_uplift.csv')


if __name__ == '__main__':
    # 1. Read in data
    CRITEO_FORMAT = load_data.DATA_FORMAT
    file = './datasets/criteo1/criteo1_1to1.csv'
    #file = './datasets/criteo_100k.csv'
    data = load_data.DatasetCollection(file, CRITEO_FORMAT)
    # 2. Train and evaluate model
    main(data)
