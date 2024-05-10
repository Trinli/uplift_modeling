"""
This version is for testing skip connections for treatment.
"""
import copy
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import src.data.load_data as load_data
import src.models.uplift_neural_net as uplift_neural_net
import itertools

if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'


class SkipNeuralNet(uplift_neural_net.UpliftNeuralNet):
    """
    This network provides the treatment label at some
    predefined inner layer of the network. The idea
    is that the network could learn an inner representation
    of "types of samples" and then with the treatment
    label decide which types of samples benefit from
    treatment.

    This class implements both the "skip neural net" with treatment
    labels passed only to one skip-layer as well as the "cumulative
    skip neural net" where the treatment label is passed as such to
    multiple layers.
    """
    def __init__(self, n_features, n_hidden_units=128, dropout_rate=.5, skip_layers=[1]):
        """
        Args:
        n_features (int): Number of input features (not counting treatment
         label).
        n_hidden_units (int): Number units in hidden layers.
        dropout_rate (float): Dropout rate for dropout layers.
        skip_layers ([int]): List of ints. Defines what layers treatment
         label is connected to (e.g. [0] causes treatment label to be fed
         into input layer, [1, 3] causes treatment label to be fed to both
         first and third hidden layers).
        """
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.dropout_rate = dropout_rate
        self.skip_layers = skip_layers
        tmp_model = self.set_model_architecture()
        super().__init__(tmp_model)


    def set_model_architecture(self):
        model = nn.ModuleList()
        # nn.EmbeddingBag(n_features, n_hidden_units, mode='sum', sparse=True),
        if self.skip_layers[0] == 0:
            # '+1' for "skip connection". Treatment feature is included here.
            model.append(nn.Linear(
                self.n_features + 1, self.n_hidden_units))
        else:
            model.append(nn.Linear(
                self.n_features, self.n_hidden_units))
        model.append(nn.Sigmoid())
        model.append(nn.Dropout(self.dropout_rate))
        model.append(nn.BatchNorm1d(self.n_hidden_units))
        for i in range(1, 3):
            if i in self.skip_layers:
                model.append(nn.Linear(
                    self.n_hidden_units + 1, self.n_hidden_units))
            else:
                model.append(nn.Linear(
                    self.n_hidden_units, self.n_hidden_units))
            model.append(nn.Sigmoid())
            model.append(nn.Dropout(self.dropout_rate))
            model.append(nn.BatchNorm1d(self.n_hidden_units))
        # The last hidden layer needs to be treated differently as the
        # output dimension should be 1:
        if 3 in self.skip_layers:
            model.append(nn.Linear(self.n_hidden_units + 1, 1))
        else:
            model.append(nn.Linear(self.n_hidden_units, 1))
        model.append(nn.Sigmoid())
        model.to(DEVICE)
        return model


    @classmethod
    def init_grid_search(cls, data,
                         n_hidden_units=[128],
                         dropout_rates=[.5],
                         skip_layers=[[0], [0, 1]]):
        """
        An alternative initialization method of this class that
        does model selection over defined hyperparameter space.
        """
        # Set initial values
        best_ecrr = 0.0  # This is the smallest possible value.
        best_model = None
        n_features = data['training_set_2']['X'].shape[1]

        # Prepare datasets:
        training_set = load_data.DatasetWrapper(data['training_set_2'])
        early_stopping_set = load_data.DatasetWrapper(data['validation_set_2a'])
        validation_set = load_data.DatasetWrapper(data['validation_set_2b'])
        for n_hidden_unit, dropout_rate, skip_layer in\
            itertools.product(n_hidden_units, dropout_rates,
                              skip_layers):
            print("\nTraining skip-model with {} ".format(n_hidden_unit) +
                  "hidden units, \ndropout rate {}, ".format(dropout_rate) +
                  "and skip layers ", end='')
            print(skip_layer)
            tmp_model = cls(n_features, n_hidden_unit,
                            dropout_rate, skip_layer)
            tmp_model.fit(training_set, early_stopping_set)
            tmp_ecrr = tmp_model.get_metrics(validation_set)
            if tmp_ecrr > best_ecrr:
                best_model = copy.deepcopy(tmp_model)
                best_ecrr = tmp_ecrr
        return best_model

    def forward(self, x, t):
        """
        Method implementing the forward pass required by torch.

        Args:
        x (torch.Tensor): The data
        t (torch.Tensor): Group label ('1' for treatment-group, '0'
         for control).
        """
        for i, layer in enumerate(self.model):
            # Only linear layers can be skip-layers.
            # Second linear layer corresponds to index 4,
            # third linear layer to 8, fourth to 12.
            if i / 4 in self.skip_layers:
                x = layer(torch.cat((x, t.view(-1, 1)), dim=1))
            else:
                x = layer(x)
        return x

    def estimate_loss(self, batch):
        """
        Method used in generic training loop (fit) in parent
        class.

        Args:
        batch (dict): A batch as returned by the DataLoader iterator.
        """
        self.set_training_mode(True)
        y_pred = self(batch['X'].to(DEVICE), batch['t'].to(DEVICE))
        loss = self.loss_fn(y_pred, batch['y'].view(-1, 1).to(DEVICE))
        return loss
