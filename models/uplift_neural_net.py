"""
Parent class to all DSL-related neural network approaches. The parent
class implements a straight forward version where the inputs are
fed as is and as interaction variables. The interaction variable
is a per-sample variable that takes values x_i when t_i == 1.
Input: x_i and t_i * x_i

In contrast to the original DSL (Lasso!), we mostly use L2-penalty.
It could make sense to use L1-penalty on the input layer, but after
that, e.g. connecting two separate networks and imposing a penalty
on the connections from one, it really does not impose sparsity in
the same sense as the original Lasso (L1-regularized linear
regression).

There is also a class for direct gradient optimization that overwrites
a lot of the functions in the parent class.
"""

import copy
import itertools
import time
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import data.load_data as load_data
import metrics.uplift_metrics as uplift_metrics


if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
BATCH_SIZE = 20000
#BATCH_SIZE = 10000
LEARNING_RATE = 0.001
#NUM_WORKERS = 16
NUM_WORKERS = 4
N_EPOCHS = 1000
#N_EPOCHS = 2
if N_EPOCHS < 100:
    warnings.warn("N_EPOCHS is set to {} for testing".format(N_EPOCHS) +
                  " purposes. It should preferably be set to > 1000.")

# Both Adagrad and Adamax would be acceptable optimizers
# (although Adamax does not support sparse input).
OPTIMIZER = torch.optim.Adagrad  # torch.optim.Adamax

# Both BCE and MSE would be acceptable loss-functions, except for direct gradient method.
# LOSS_FN = torch.nn.BCELoss
LOSS_FN = torch.nn.MSELoss

class UpliftNeuralNet(nn.Module):
    """
    PARENT CLASS for neural network approaches to uplift modeling.
    This class does not contain an own model and is not complete
    without additional methods. See class DSLNeuralNet below for
    example.

    Notes:
    Methods that should remain as is are:
    -__init__() (create own __init__ method for class and call
     super().__init__().
    -fit()
    -predict_uplift()
    -get_metrics()

    When making a new class that inherits this, changes might be
    needed to the following:
    1. small additions to __init__() (e.g. storage of previously
     undefined variables, still call super().__init__(), though).)
    2. set_model_architecture() should create your _new_ neural net
    3. forward() needs to work with model architecture
    4. set_training_mode() for the new architecture
    5. estimate_loss() for the new architecture.
    6. optional: init_grid_search(), initialization using grid
     search to find optimal model in grid.
    """

    def __init__(self, model):
        """
        Args:
        model: A torch-compatible neural network.
        """
        super().__init__()
        self.ecrr = 0  # Expected conversion rate over all treatment rates
        self.model = model
        self.optimizer = OPTIMIZER(self.model.parameters(),
                                   lr=LEARNING_RATE)
        self.loss_fn = LOSS_FN()

    def set_training_mode(self, mode=False):
        """
        Method for setting model to training or evaluation model
        Necessary for dropout. The need for this method is apparent with
        more advanced models not supported by nn.Sequential.

        Args:
        mode (bool): True indicates training mode, False evaluation
         mode.
        """
        self.model.train(mode)

    def fit(self, training_data,
            validation_data,
            n_epochs=N_EPOCHS,
            max_epochs_without_improvement=100,
            verbose=False):
        """
        Function for fitting (training) defined neural net. The progress of
        training is measured based on the expected conversion rate for the
        treatment plan given no prior information on preference on fraction
        of treated samples (uniform prior). This is equivalent to some constant
        plus AUUC. See e.g. Jaskowski & Jaroszewicz (2012) for more details
        on metric.
        Training stops if some of the set criteria are met, or if training
        time exceeds 23 hours.

        Args:
        training_data (load_data.DatasetWrapper): Data used for training
         neural network.
        validation_data (load_data.DatasetWrapper): Data used for
         evaluation of training
         progress and early stopping.
        n_epochs (int): Maximum number of epochs
        max_epochs_without_improvement (int): Criterion for stopping training
         if a better model as evaluated on validation set has not been found
         after this many iterations.
        verbose (bool): Setting this to True will lead to slightly more output
         during training.
        """
        start_time = time.time()
        max_training_time = 22 * 60 * 60  # Seconds in 22 hours.
        epochs_without_improvement = 0
        best_model = self.model
        tmp_training_dataloader = DataLoader(
            training_data, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)
        for i in range(n_epochs):
            if time.time() - start_time > max_training_time:
                # If training time has exceeded set limit.
                print("Maximum training time reached ({} seconds)".format(max_training_time))
                break
            if i % 10 == 0:
                # Estimate goodness of current model:
                val_ecrr = self.get_metrics(validation_data)
                print("Expected conversion rate (ECRR): {}".format(
                    val_ecrr))
                if self.ecrr < val_ecrr:
                    self.ecrr = val_ecrr
                    # Store new best model and reset counter
                    best_model = copy.deepcopy(self.model)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 10
                    if epochs_without_improvement > \
                       max_epochs_without_improvement:
                        print("No improvement during last {} ".format(
                            max_epochs_without_improvement) +
                              "epochs after {} epochs.".format(i))
                        # Stop training:
                        break
            # Actual training code:
            for batch in tmp_training_dataloader:
                loss = self.estimate_loss(batch)
                if verbose:
                    # Not tested:
                    print("Training set loss: {}".format(loss))
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
        if best_model is self.model:
            # Sanity check
            print("Model not updated. No improvement found in " +
                  "{} epochs.".format(i))
        # When done, set self.model to best model:
        training_time = time.time() - start_time
        training_hours = training_time / (60 * 60)
        print("Training time: {} hours".format(training_hours))
        self.model = best_model

    def predict_uplift(self, dataset):
        """
        Method for predicting uplift using the classifier. This should
        be fairly generic for different model architectures as long as
        forward() and set_training_mode() are defined.

        Args:
        dataset (DatasetWrapper): Format from load_data.py.

        Note:
        Could potentially be changed to take only 'X' as input? That would be
         in line with how the non-neural net based uplift models work.
        """
        # When predicting, model mode always needs to be set to false.
        self.set_training_mode(False)
        tmp_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS)
        uplift_predictions = None
        with torch.autograd.no_grad():
            for batch in tmp_dataloader:
                control_predictions = self(
                    batch['X'].to(DEVICE),
                    torch.zeros(batch['t'].size()).to(DEVICE)).cpu()
                treatment_predictions = self(
                    batch['X'].to(DEVICE),
                    torch.ones(batch['t'].size()).to(DEVICE)).cpu()
                tmp_uplift_predictions = treatment_predictions - \
                    control_predictions
                if uplift_predictions is None:
                    uplift_predictions = tmp_uplift_predictions
                else:
                    uplift_predictions = torch.cat(
                        (uplift_predictions, tmp_uplift_predictions))
            uplift_predictions = uplift_predictions.detach().numpy()
            # Change to vector:
            uplift_predictions = uplift_predictions.reshape((-1, ))
        return uplift_predictions

    def get_metrics(self, dataset):
        """
        Method for estimating expected conversion rate for dataset.
        This method does the prediction step and wraps the
        functions from uplift_metrics.py.

        Args:
        dataset (load_data.DatasetWrapper): Dataset to estimate the
        metrics (for presumably validation set).
        """
        self.set_training_mode(False)
        uplift_predictions = self.predict_uplift(dataset)
        # The type changes necessary for pytorch need to be undone:
        ecrr = uplift_metrics.auuc_metric(
            data_class=dataset[:]['y'].astype(bool),
            data_score=uplift_predictions,
            data_group=dataset[:]['t'].astype(bool),
            ref_plan_type='zero')
        return ecrr


class DSLNeuralNet(UpliftNeuralNet):
    """
    Class that implements a DSL-version of neural nets
    """
    def __init__(self,
                 n_features,
                 n_hidden_units=128,
                 dropout_rate=.5,
                 dsl_penalty=.001):
        """
        Args:
        n_features (int): Number of features in data
        n_hidden_units (int): Number of units in hidden layers
        dropout_rate (float): Drop-out rate for drop-out layers.
         Falls within [0, 1[
        dsl_penalty (float): Penalty to use for connecting interaction
         network to remaining net.
        """
        self.n_features = n_features
        self.dsl_penalty = dsl_penalty
        self.n_hidden_units = n_hidden_units
        self.dropout_rate = dropout_rate
        tmp_model = self.set_model_architecture()
        super().__init__(tmp_model)

    def set_model_architecture(self):
        """
        Method for setting model architecture.
        """
        model = nn.Sequential(
            # Alternative input layer for embeddings:
            # nn.EmbeddingBag(n_features, n_hidden_units, mode='sum',
            # sparse=True),
            # '2 * n...' to include space for interaction variables
            nn.Linear((2 * self.n_features), self.n_hidden_units),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(self.n_hidden_units),

            nn.Linear(self.n_hidden_units, self.n_hidden_units),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(self.n_hidden_units),

            nn.Linear(self.n_hidden_units, self.n_hidden_units),
            # nn.ReLU(),
            nn.Sigmoid(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(self.n_hidden_units),

            nn.Linear(self.n_hidden_units, 1),
            nn.Sigmoid(),
        ).to(DEVICE)
        return model

    @classmethod
    def init_grid_search(cls,
                         data,
                         n_hidden_units=[128],
                         dropout_rates=[.5],
                         # =[0.1, 0.3, .01, .003, .001, .0003, .0001]):
                         dsl_penalties=[.003, .001]):
        """
        Alternative initialization method that does model
        selection over hyperparameters. Early stopping is
        done based on validation_set_2a and the model selection
        based on validation_set_2b (both are contained in data).
        The initialization method returns the best model based
        on validation set performance.

        Args:
        data (load_data.DatasetCollection): Data with training set,
         two validation sets, and testing set.
        """
        # ECRR: Expected conversion rate over r (all treatment rates)
        # This metric is auuc for treatment plan + a constant.
        best_ecrr = 0
        best_model = None
        n_features = data['training_set']['X'].shape[1]

        # Training set to fit models:
        training_set = load_data.DatasetWrapper(data['training_set_2'])
        # Validation set (1) for early stopping:
        early_stopping_set = \
            load_data.DatasetWrapper(data['validation_set_2a'])
        # Validation set (2) for model selection:
        validation_set = load_data.DatasetWrapper(data['validation_set_2b'])

        # Iterate over all combinations of hidden units, dropout
        # rates, and dsl-penalties:
        for n_hidden_unit, dropout_rate, dsl_penalty in\
            itertools.product(n_hidden_units, dropout_rates, dsl_penalties):
            print("\nTraining vanilla DSL model " +
                  "with {} hidden units in each".format(n_hidden_unit) +
                  " layer, \ndropout rate of {}".format(dropout_rate) +
                  " and dsl-penalty {}.".format(dsl_penalty))
            tmp_model = cls(n_features, n_hidden_unit,
                            dropout_rate, dsl_penalty)
            tmp_model.fit(training_set, early_stopping_set)
            tmp_ecrr = tmp_model.get_metrics(validation_set)
            if tmp_ecrr > best_ecrr:
                best_model = copy.deepcopy(tmp_model)
                best_ecrr = tmp_ecrr
        return best_model

    def forward(self, x, t):
        """
        Forward-pass (read pytorch documentation for info). Used only
        for fitting. Uplift predictions is done with two prediction
        phases with slightly different data (see predict_uplift()).

        Args:
        x (torch.Tensor): The data
        t (torch.Tensor): Group label ('1' for treatment-group, '0'
         for control).
        """
        return self.model(torch.cat([x, x * \
            t.view(-1, 1).repeat(1, x.size(1))], dim=1))

    def estimate_loss(self, batch):
        """
        Method that allows the fit-method to remain constant over multiple
        different models. Here we use the L2-norm of the connecting weights
        for penalty.

        Args:
        batch (dict): A batch as returned by the DataLoader iterator.
        """
        self.set_training_mode(True)
        y_pred = self(batch['X'].to(DEVICE), batch['t'].to(DEVICE))
        loss = self.loss_fn(y_pred, batch['y'].view(-1, 1).to(DEVICE)) +\
            self.dsl_penalty * torch.norm(
                   self.model[0].weight[:, self.n_features:] *
                   torch.mean(batch['t'].to(DEVICE).float()), 2)
        # The penalty for 't' samples simply ends up being the mean of
        # 't' in one batch times the L2-norm of the weight.
        # Set to prediction mode:
        self.set_training_mode(False)
        return loss





class DirectGradientUpliftNN(UpliftNeuralNet):
    """
    Class that implements a neural net for uplift modelign that
    directly minimizes E(MSE) on training set, i.e. predicts tau
    and minimizes the squared difference between tau and the
    revert label.
    """
    def __init__(self,
                 n_features,
                 n_hidden_units=128,
                 dropout_rate=.5):
        """
        Args:
        n_features (int): Number of features in data
        n_hidden_units (int): Number of units in hidden layers
        dropout_rate (float): Drop-out rate for drop-out layers.
         Falls within [0, 1[
        """
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.dropout_rate = dropout_rate
        tmp_model = self.set_model_architecture()
        super().__init__(tmp_model)
        # OVERRIDE super()'s cost function. Needs to be one that outputs in [-1, 1].

    def set_model_architecture(self):
        """
        Method for setting model architecture.
        """
        model = nn.Sequential(
            # Alternative input layer for embeddings:
            # nn.EmbeddingBag(n_features, n_hidden_units, mode='sum',
            # sparse=True),
            nn.Linear(self.n_features, self.n_hidden_units),
            nn.LeakyReLU(),
            # nn.Sigmoid(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(self.n_hidden_units),

            nn.Linear(self.n_hidden_units, self.n_hidden_units),
            nn.LeakyReLU(),
            # nn.Sigmoid(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(self.n_hidden_units),

            nn.Linear(self.n_hidden_units, self.n_hidden_units),
            nn.LeakyReLU(),
            # nn.Sigmoid(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(self.n_hidden_units),

            nn.Linear(self.n_hidden_units, self.n_hidden_units),
            nn.LeakyReLU(),
            # nn.Sigmoid(),
            nn.Dropout(self.dropout_rate),
            nn.BatchNorm1d(self.n_hidden_units),

            nn.Linear(self.n_hidden_units, 1),
            # nn.Sigmoid(),
            nn.Tanh(),  # Tanh naturally falls within [-1, 1]  - just the thing needed!
        ).to(DEVICE)
        return model

    # FIX: THE OUTPUT IS INDEPENDENT OF T
    def forward(self, x):
        """
        Forward-pass (read pytorch documentation for info). Used only
        for fitting. Uplift predictions is done with two prediction
        phases with slightly different data (see predict_uplift()).

        Args:
        x (torch.Tensor): The data
        t (torch.Tensor): Group label ('1' for treatment-group, '0'
         for control).
        """
        return self.model(x)
        #return self.model(torch.cat([x, x * \
        #    t.view(-1, 1).repeat(1, x.size(1))], dim=1))

    # NEED A NEW PREDICT_UPLIFT-function, overwrite
    def predict_uplift(self, dataset):
        """
        Method for predicting uplift using the classifier. 
        The direct uplift model predicts tau instead of using
        two models or two predictions.

        Args:
        dataset (DatasetWrapper): Format from load_data.py.

        Note:
        Could potentially be changed to take only 'X' as input? That would be
         in line with how the non-neural net based uplift models work.
        """
        # When predicting, model mode always needs to be set to false.
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


    def estimate_loss(self, batch):
        """
        Method that allows the fit-method to remain constant over multiple
        different models. Here we use the L2-norm of the connecting weights
        for penalty.

        Args:
        batch (dict): A batch as returned by the DataLoader iterator.
        """
        self.set_training_mode(True)
        tau_pred = self(batch['X'].to(DEVICE))
        loss = self.loss_fn(tau_pred, batch['r'].view(-1, 1).to(DEVICE).float())
        # Set to prediction mode:
        self.set_training_mode(False)
        return loss
