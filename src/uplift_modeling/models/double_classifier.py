"""
Code for double classifier approaches needed for tests.
"""
from uplift_modeling.load_data import DatasetWrapper
import uplift_modeling.models.neural_net as neural_net
from numpy import where
import warnings

import torch
from torch import nn

# Number of epochs for neural network training
N_EPOCHS = 1000
if N_EPOCHS < 100:
    warnings.warn("N_EPOCHS is set to {}. It should ".format(N_EPOCHS) +
                  "preferably be >= 1000.")


class DoubleClassifier(object):
    """
    Parent class for all double-classifier approaches for uplift modeling.
    """
    def __init__(self, base_learner, *args, **kwargs):
        """
        Initialization for a double classifier.

        Args:
        base_learner (class): Base-learner to use for double classifier approach.
         Requires the base_learner to have methods fit() and predict() to function
         correctly.
        """
        self.treatment_model = base_learner(*args, **kwargs)
        self.control_model = base_learner(*args, **kwargs)

    def fit(self, data, undersampling=None, **kwargs):
        self.treatment_model.fit(data['training_set', undersampling, 'treatment']['X'],
                                 data['training_set', undersampling, 'treatment']['y'], **kwargs)
        self.control_model.fit(data['training_set', undersampling, 'control']['X'],
                               data['training_set', undersampling, 'control']['y'], **kwargs)

    def predict_uplift(self, X):
        treatment_tmp = self.treatment_model.predict(X)
        control_tmp = self.control_model.predict(X)
        uplift_predictions = treatment_tmp - control_tmp
        uplift_predictions = uplift_predictions.reshape((-1, ))
        return uplift_predictions

    def set_train(self, mode=True):
        """
        Only applicable for neural nets.
        """
        self.treatment_model.train(mode=mode)
        self.control_model.train(mode=mode)


class LogisticRegression(DoubleClassifier):
    """
    Double classifier using logistic regression
    Overwriting the binary classification model from sklearn here.
    """
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        super().__init__(LogisticRegression)

    def predict_uplift(self, X):
        treatment_tmp = self.treatment_model.predict_proba(X)
        control_tmp = self.control_model.predict_proba(X)
        # The lr-models produce probabilities for both classes.
        # Only one class is the "positive" one we want to predict.
        # Rather ugly way to extract the right index...
        t_idx = where(self.treatment_model.classes_ == 1.0)[0][0]
        c_idx = where(self.control_model.classes_ == 1.0)[0][0]
        uplift_predictions = treatment_tmp[:, t_idx] - control_tmp[:, c_idx]
        return uplift_predictions


class SequentialTorch(torch.nn.Module):
    """ A wrapper providing fitting and predictions for a torch model. """

    def __init__(self, model, 
                 optimizer=torch.optim.RMSprop, lr=0.005, 
                 niter=max(N_EPOCHS, 10000), noimprov_niter=100): 
        super().__init__()
        self.model = model

        self._optimizer = optimizer
        self._lr = lr
        self._niter = niter
        self._noimprov_niter = noimprov_niter

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, requires_grad=False)

        out = torch.softmax(self.model(x), -1)
        return out

    def predict(self, x):
        return self.forward(x)[:, 1]

    def fit(self, inputs, labels, verbose=True, batch_size=128, **kwargs):
        if not torch.is_tensor(inputs):
            inputs = torch.tensor(inputs, requires_grad=False)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels, dtype=torch.long, requires_grad=False)  

        dataset = torch.utils.data.TensorDataset(inputs, labels) 
        loader = torch.utils.data.DataLoader(dataset, pin_memory=True, batch_size=batch_size)
        return self.fit_loader(loader, verbose=verbose,  **kwargs)

    def fit_loader(self, loader, verbose=True, nepochs=100, noimprov_niter=1000):
        self.set_train(mode=True)

        optimizer = self._optimizer(self.model.parameters(), lr=self._lr)
        criterion = torch.nn.CrossEntropyLoss()
        loss = lambda x_y: criterion(self(x_y[0]), x_y[1])
        print_iter = lambda epoch, i, loss, model: \
         (print('[SequentialTorch.fit][%d] loss: %.3f' % (i, loss)) \
          if verbose and i%100==0 else None)
        torch.fit(self, loader, optimizer, loss, 
                      nepochs=nepochs, noimprov_niter=noimprov_niter,
                      callback = print_iter)

    def set_train(self, mode=True):
        self.model.train(mode=mode)


class IndividualLogisticRegressionPyTorch(SequentialTorch):
    """ A replacement for LogisticRegression from sklearn.linear_model.

        Can be used along with DoubleClassifier as follows:
        DoubleClassifier(IndividualLogisticRegressionPyTorch, n_features=...).  
    """

    def __init__(self, n_features, output_dim=2, **kwargs): 
        model = torch.nn.Linear(n_features, output_dim)
        super().__init__(model, **kwargs)


class IndividualNNPyTorch(SequentialTorch):
    """Neural network.

    Can be used instead of LogisticRegression with DoubleClassifier.
    See IndividualLogisticRegressionPyTorch.
    """

    def __init__(self, n_features, output_dim=2, 
                       n_hidden_units=128, dropout_rate=.5, dependent_var='y', 
                       **kwargs):
        model = torch.nn.Sequential(
            # nn.EmbeddingBag(n_features, n_hidden_units, mode='sum', sparse=True),
            nn.Linear(n_features, n_hidden_units),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(n_hidden_units),

            nn.Linear(n_hidden_units, n_hidden_units),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(n_hidden_units),

            nn.Linear(n_hidden_units, n_hidden_units),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(n_hidden_units),

            nn.Linear(n_hidden_units, output_dim),
        )
        super().__init__(model, **kwargs)


class SupportVectorMachine(DoubleClassifier):
    """Double classifier using SVM and isotonic regression

    """

    class SVM_IR(object):
        """
        Auxiliary class for support vector machine and isotonic regression.
        The classifier is a pipeline of two where a support vector machine
        first predicts a score and isotonic regression further transforms
        this into a probability.
        """

        def __init__(self):
            from sklearn.svm import SVC
            from sklearn.isotonic import IsotonicRegression
            self.svm_model = SVC()
            self.ir_model = IsotonicRegression(y_min=0.001, y_max=0.999,
                                               out_of_bounds='clip')

        def fit(self, training_set, calibration_set):
            # HYPERPARAMETERS?!? USE DEFAULT?
            self.svm_model.fit(X=training_set['X'], y=training_set['y'])
            tmp_cal = self.svm_model.decision_function(X=calibration_set['X'])
            self.ir_model.fit(tmp_cal, calibration_set['y'])

        def predict(self, X):
            tmp_score = self.svm_model.decision_function(X)
            tmp_prob = self.ir_model.predict(tmp_score)
            return tmp_prob

    def __init__(self):
        super().__init__(SupportVectorMachine.SVM_IR)

    def fit(self, data):  # , undersampling=None): # Quick fix
        """
        Method for fitting the SVM-IR combination for double classifier approach.
        This had to be overwritten as fitting requires both training set and
        calibration set.

        Args:
        undersampling (str): {None, '11', '1111'}. Same as in parent class. Note
         that only the training set is undersampled, not the calibration set. The
         calibration needs to be done on data with natural sampling rate.
        """
        undersampling = '1111'
        self.treatment_model.fit(data['training_set', undersampling, 'treatment'],
                                 data['validation_set', None, 'treatment'])
        self.control_model.fit(data['training_set', undersampling, 'control'],
                               data['validation_set', None, 'control'])


class DCNeuralNet(DoubleClassifier):
    """
    A class for using neural networks with the double classifier approach.
    Wonder how this will work...
    """
    def __init__(self, n_features, **kwargs):
        """
        Initialize

        Args:
        base_learner (class): The base learner needs to be a model that can
         be trained and is compatible with the approach used for other neural
         networks here.
        """
        # Neural net needs info on features for initialization.
        self.n_features = n_features
        # self.dependent_var = dependent_var
        self.treatment_model = neural_net.NeuralNet(n_features=self.n_features, **kwargs)
        self.control_model = neural_net.NeuralNet(n_features=self.n_features, **kwargs)


    def fit(self, data, undersampling=None, n_epochs=N_EPOCHS):  # Undersampling not implemented
        # for training_set_2 and validation_sets
        """

        Args:
        data (?): Contains both training set and validation set for early stopping.
         Also contains a second validation set for model selection.
        undersampling (str): {None, '11', '1111'}. Only training data is undersampled.
         Validation set is not as we want best performance on data with natural
         sampling rate.
        hyperparameters (?): ???
        """
        treatment_train = DatasetWrapper(data['training_set_2', undersampling,
                                              'treatment'])
        treatment_val = DatasetWrapper(data['validation_set_2a', None, 'treatment'])
        # Split between 'X' and 'y' needs to be done _after_ batching, i.e.
        # in training loop for neural net. Perhaps pass 'dependent var'?
        self.treatment_model.fit(treatment_train, treatment_val, n_epochs=n_epochs)
        control_train = DatasetWrapper(data['training_set_2', undersampling,
                                            'control'])
        control_val = DatasetWrapper(data['validation_set_2a', None, 'control'])
        self.control_model.fit(control_train, control_val, n_epochs=n_epochs)


class DCRandomForest(DoubleClassifier):
    """
    Double-classifier approach with random forest.
    """
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.treatment_model = RandomForestClassifier()
        self.control_model = RandomForestClassifier()

    def fit(self, data):
        """
        Method for fitting the double classifier

        Args:
        X (numpy.array): Features
        y (numpy.array): Labels
        """
        treatment_data = data['training_set', None, 'treatment']
        control_data = data['training_set', None, 'control']
        self.treatment_model.fit(X=treatment_data['X'], y=treatment_data['y'])
        self.control_model.fit(X=control_data['X'], y=control_data['y'])

    def predict_uplift(self, X):
        """
        Method for predicting uplift.

        Args:
        X (numpy.array): Features
        """
        t_prob = self.treatment_model.predict_proba(X)
        c_prob = self.control_model.predict_proba(X)
        t_idx = where(self.treatment_model.classes_ == 1.0)[0][0]
        c_idx = where(self.control_model.classes_ == 1.0)[0][0]
        uplift_predictions = t_prob[:, t_idx] - c_prob[:, c_idx]
        return uplift_predictions
