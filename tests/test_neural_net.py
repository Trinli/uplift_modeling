"""
Script for testing neural net code.
"""

import data.load_data as load_data
import models.neural_net as neural_net


def test():
    data = load_data.DataSet("./datasets/criteo_10k.csv")
    model = neural_net.NeuralNet(data.nro_features)
    model.train(data['training_set', 'all', 'dataloader'], data['validation_set', 'all', 'dataloader'])
    return model


if __name__ == '__main__':
    test()
