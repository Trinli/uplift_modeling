"""
Code for residual uplift models. When training a residual model,
we first train a response model on the subset of training data
belonging to the control group. Using the resulting model, we
then predict the class for the subset of training data belonging
to the treatment group. We thus get a counterfactual prediction
for these samples had the treatment not been applied. We calculate
the _residual_, i.e. the difference between actual outcome (with
treatment) and the counterfactual. Then we train a model using
the residual as dependent variable.
Using the resulting model, we can directly predict the uplift
with only one model.

Note that the residual falls in [-1, 1] and hence the second model
must be a regression model, not a classification model, unless we
extend some classification method to predict in [-1, 1].
"""

import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import LinearRegression
import uplift_modeling.data.load_data as load_data
import uplift_modeling.models.neural_net as neural_net
import uplift_modeling.models.uplift_neural_net as uplift_neural_net

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
BATCH_SIZE = 200000
NUM_WORKERS = 8

N_EPOCHS = 1000
if N_EPOCHS < 100:
    warnings.warn("N_EPOCHS set to {} in residual_neural_net.py ".format(N_EPOCHS) +
                  "for testing purposes. Should preferably be > 100.")


class ResidualUpliftModel(object):
    """
    Parent class for residual-approaches for
    uplift modeling.
    """
    def __init__(self, base_learner):
        """
        Args:
        base_learner: base learner to use for modeling,
         e.g. sklearn.linear_model.LinearRegression.
         Note that the base learner must be a regression
         model, not a classification model, as the same
         base learner is used both for the response model
         and the residual model.

        Notes:
        This class could be changed to take two base learners, one
        for the classification task, and another for the regression.
        """
        # First create a model that will predict the response
        # without treatment:
        self.response_model = base_learner()
        # Second, create a model that will predict the difference
        # between the predicted response _without_ treatment and
        # the actual response _with_ treatment.
        self.residual_model = base_learner()

    def fit(self, data, undersampling=None):
        """
        Method for fitting the residual model.

        Args:
        data (load_data.DatasetCollection): Object containing
         the data.
        undersampling {None, '11', '1111'}: Defines whether
         undersampling should be used training. '11' indicates
         1:1 treatment to control sample ratio. '1111' indicates
         1:1:1:1 positive treatment to negative treatment to
         positive control to negative control samples.
        """
        # Train response model:
        response_data = data['training_set', undersampling, 'control']
        self.response_model.fit(X=response_data['X'], y=response_data['y'])
        # Predict response without treatment for treated samples:
        treatment_data = data['training_set', undersampling, 'treatment']
        predicted_response = self.response_model.predict(treatment_data['X'])
        # Estimate difference between prediction (without treatment) and
        # actual outcome (with treatment):
        residual = treatment_data['y'] - predicted_response
        # Train residual model:
        self.residual_model.fit(X=treatment_data['X'], y=residual)

    def predict_uplift(self, X):
        """
        Method for predicting uplift. Note that this requires
        using only the second model (residual_model).

        Args:
        X (numpy.array([...])): features of your data
        """
        return self.residual_model.predict(X)


class ResidualLinearRegression(ResidualUpliftModel):
    """
    Simple wrapper Class for linear regression for residual model.
    """
    def __init__(self):
        """
        Nothing to see here.
        """
        super().__init__(LinearRegression)


class ResidualNeuralNet(ResidualUpliftModel):
    """
    Residual uplift model with neural networks. The response
    model is implemented as a classification model and the
    second (residual) model is implemented as a regression
    model.
    """
    def __init__(self, n_features):
        """
        Args:
        n_features (int): Number of features in data.
        """
        # Response model can be either classifier or regression model.
        # The response model should be optimized based on AUC-ROC or
        # similar classification metric.
        self.response_model = neural_net.NeuralNet(n_features)
        # The residual model has to be a regression model.
        # The residual model should preferably be optimized based
        # on ECRR (or other uplift metric).
        self.residual_model = RegressionNeuralNet(n_features)


    class ResidualDatasetWrapper(Dataset):
        """
        Auxiliary class for ResidualNeuralNet. This class
        allows you to define the contents of the dependent
        variable at runtime.
        This inherits the Dataset-class from torch which is
        suitable input for the DataLoader necessary for
        training.
        """
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __getitem__(self, idx):
            return {'X': self.X[idx, :],
                    'y': self.y[idx]}

        def __len__(self):
            return self.X.shape[0]


    def fit(self, data, undersampling=None):
        """
        Method for fitting both response and residual models.

        Args:
        data (load_data.DatasetCollection): Object containing
         the data.
        undersampling {None, '11', '1111'}: Defines whether
         undersampling should be used training. '11' indicates
         1:1 treatment to control sample ratio. '1111' indicates
         1:1:1:1 positive treatment to negative treatment to
         positive control to negative control samples.
        """
        # Train response model:
        response_data = load_data.DatasetWrapper(
            data['training_set', undersampling, 'control'])
        val_response_data = load_data.DatasetWrapper(
            data['validation_set', None, 'control'])
        self.response_model.fit(response_data, val_response_data,
                                n_epochs=N_EPOCHS)

        # Predict response without treatment for treated samples:
        treatment_data = data['training_set', undersampling, 'treatment']
        predicted_response = self.response_model.predict(treatment_data['X'])
        predicted_response = predicted_response.reshape((-1, ))

        # Estimate difference between prediction (without treatment) and
        # actual outcome (with treatment):
        residual = treatment_data['y'] - predicted_response

        # Format this with a purpose built dataset wrapper:
        residual_dataset = ResidualNeuralNet.ResidualDatasetWrapper(
            treatment_data['X'], residual)
        # As the residual model should be optimized based on ECRR,
        # the validation set needs to contain samples of both
        # groups and represent natural sampling rate.
        validation_set = load_data.DatasetWrapper(
            data['validation_set', None, 'all'])

        # Train residual model (maximize validation set ECRR)
        self.residual_model.fit(residual_dataset, validation_set,
                                n_epochs=N_EPOCHS)

    def predict_uplift(self, X):
        """
        Method to predict uplift from data. The uplift is in a
        residual model modeled directly with the second model,
        hence predicting using only that produces directly the
        change in conversion probability (=uplift).

        Args:
        X (numpy.array([float])): Array of data for prediction.
        """
        return self.residual_model.predict(X)


class RegressionNeuralNet(uplift_neural_net.UpliftNeuralNet):
    """
    When training a residuan neural net, you need both a classifier
    to predict the response for control samples and a regression model
    to predict the residual. This is that regression model.

    This class does not actually have the
    same structure as the parent class, but some concepts are borrowed
    from there (fit(), set_training_mode(), and get_metrics()). These
    directly optimize for ECRR on the validation set.
    -Perhaps replace with neural_net.NeuralNet?
    -It would make sense to have a vanilla neural net for classification
    (neural_net.NeuralNet) and another vanilla neural net for regression.
    """
    def __init__(self, n_features, n_hidden_units=128, dropout_rate=.5):
        """
        Args:
        n_features (int): Number of input features per sample.
        n_hidden_units (int): Number of units per hidden layer.
        dropout_rate (float): Dropout rate for dropout layers.
        """
        self.n_hidden_units = n_hidden_units
        self.dropout_rate = dropout_rate
        tmp_model = torch.nn.Sequential(
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

            nn.Linear(n_hidden_units, 1),
            nn.Sigmoid(),
        ).to(DEVICE)

        super().__init__(tmp_model)


    def forward(self, X):
        """
        Used by predict_uplift().
        X (?): X from batch as provided by DataLoader iterator.
        """
        return self.model(X)

    def estimate_loss(self, batch):
        """
        Method used by fit() to estimate loss for batch.

        Args:
        batch (?): batch as provided by DataLoader iterator.
        """
        y_pred = self.model(batch['X'].to(DEVICE))
        loss = self.loss_fn(y_pred, batch['y'].view(-1, 1).to(DEVICE))
        return loss

    def predict_uplift(self, dataset):
        """
        Predicts the response for a regression neural net model.
        This may OR may not be uplift. Notice that this function is
        here to satisfy the requirement for get_metrics() from the
        parent class. This works correctly in the context of the
        ResidualNeuralNet.

        Args:
        dataset (load_data.DatasetWrapper): A dataset
        """
        self.set_training_mode(False)
        tmp_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS)
        uplift_predictions = None
        with torch.autograd.no_grad():
            for batch in tmp_dataloader:
                tmp_uplift_predictions = self(
                    batch['X'].to(DEVICE)).cpu()
                if uplift_predictions is None:
                    uplift_predictions = tmp_uplift_predictions
                else:
                    uplift_predictions = torch.cat(
                        (uplift_predictions, tmp_uplift_predictions))
        uplift_predictions = uplift_predictions.detach().numpy()
        # Change to vector:
        uplift_predictions = uplift_predictions.reshape((-1, ))
        return uplift_predictions

    def predict(self, X):
        """
        Method for predicting using the model trained in this class.
        This is the uplift for the ResidualNeuralNet. This is done
        on the CPU.

        Args:
        X (numpy.array([[float, ...], ...]])): An array with your
         data (samples in rows, features in columns).
        """
        # Predicting all in one batch:
        self.set_training_mode(False)
        # Predict in one batch
        self.model.cpu()
        tmp = self.model(torch.from_numpy(X)).detach().numpy()
        tmp = tmp.reshape((-1, ))
        self.model.to(DEVICE)
        return tmp
