"""
Implementation of rever-label as proposed by Athey &
Imbens (2015). This is also called class-variable
transformation, but I picked the 'revert-label' name
to separate this from class-variable transformation
following Jaskowski & Jaroszewicz (2012).
"""
from uplift_modeling.load_data import DatasetWrapper


class RevertLabel(object):
    """
    Parent class for rever-label based approaches
    """

    def __init__(self, base_learner):
        self.model = base_learner()

    def fit(self, data):
        """
        Args:
        data (load_data.DatasetCollection)
        """
        self.model.fit(data['training_set']['X'],
                       data['training_set']['r'])

    def predict_uplift(self, X):
        return self.model.predict(X)


class RLRegression(RevertLabel):
    """
    Class for linear regression with revert label.
    """
    def __init__(self):
        from sklearn.linear_model import LinearRegression
        super().__init__(LinearRegression)


class RLNeuralNet(RevertLabel):
    """
    Class for neural net with revert label.
    """
    def __init__(self, n_features):
        # The neural net in neural_net.py does not support continuous
        # dependent variable for the moment. The value of implementing
        # this is dubious at best.
        # SKIP FOR NOW?
        from models.neural_net import NeuralNet
        raise Exception("Not implemented yet!")
        self.n_features = n_features
        # The 'r' as dependent var comes from the
        # load_data.DatasetCollection class
        self.model = NeuralNet(n_features, dependent_var='r')

    def fit(self, data):
        """
        Method for fitting neural network with revert-label.

        Args:
        data (load_data.DatasetCollection): Contains both training set and
        validation set for early stopping. Also contains a second validation
        set for model selection.
        hyperparameters (?): ???
        """
        training_data = DatasetWrapper(data['training_set_2'])
        validation_data = DatasetWrapper(data['validation_set_2a'])

        # Split between 'X' and 'z' needs to be done _after_ batching, i.e.
        # in training loop for neural net. Perhaps pass 'dependent var'?
        self.model.fit(training_data, validation_data, n_epochs=2)

    def predict_uplift(self, X):
        """
        Wrapper for predict_uplift from parent class.
        The output needs reshaping.
        """
        predicted_uplift = self.model.predict(X)
        predicted_uplift = predicted_uplift.reshape(-1).astype(float64)
        return predicted_uplift
