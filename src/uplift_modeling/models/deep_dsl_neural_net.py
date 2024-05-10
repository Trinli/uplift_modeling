"""
Deep Data-shared "Lasso" and random search in network architectures

This code is made to test a similar approach to the Data-shared Lasso
proposed by Gross & Tibshirani (2016). Note that while the Lasso uses
L1-regularization, we use L2.
Essentially we are trying a neural network which takes inputs twice
as x and t*x. The second inputs are hence interaction terms.
The intention is to do a random search for the network architecture
to perhaps find a network that outperforms everything up to
date.
"""
import random
import torch
import torch.nn as nn
import warnings
import uplift_modeling.data.load_data as load_data
import uplift_modeling.models.uplift_neural_net as uplift_neural_net


LEARNING_RATE = .1
if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

# Both Adagrad and Adamax would be acceptable optimizers
# (although Adamax does not support sparse input).
OPTIMIZER = torch.optim.Adagrad  # torch.optim.Adamax

# Both BCE and MSE would be acceptable loss-functions.
LOSS_FN = torch.nn.BCELoss  # torch.nn.MSELoss()


class DeepDslNeuralNet(uplift_neural_net.UpliftNeuralNet):
    """
    Class for deep data-shared lasso (dsl) -type of neural net.
    Essentially we first create two separate neural networks,
    one for input x and another for input t*x, then we connect
    both of these nets, with penalty on the weights coming from
    the network for input t*x, into the reminder (REM) of the
    net. The reminder then predicts the label. Think of it
    as a network starting from two branches and merging into
    one. To predict uplift, we have to make two predictions
    for x: one where t=0 and another where t=1.

    The "DSL" name is borrowed from Gross & Tibshirani (2016).
    While they used an actual Lasso (L1-regularized linear
    regression), we use L2 penalty as sparsity in the network
    itself is not useful in this context.
    """
    def __init__(self, n_features,
                 layers_dsl=6,
                 layers_rem=6,
                 hidden_units_dsl=12,
                 hidden_units_rem=12,
                 dropout_rate=.5,
                 dsl_penalty=1,
                 learning_rate=LEARNING_RATE):
        """
        Args:
        n_features (int): Number of features in data.
        layers_dsl (int): Number of layers in the dsl-forks
         of the neural net
        layers_rem (int): Number of layers in the remainder
         of the neural net after the dsl-layers
        hidden_units_dsl (int): Number of hidden units in
         every dsl-layer in the net
        hidden_units_rem (int): Number of hidden units in
         every rem-layer in the net
        dropout_rate (float): Dropout rate used in all dropout
         layers
        """
        # Initialize the nn.Module.
        # uplift_neural_net.__init__() is not useful in this context.
        super(uplift_neural_net.UpliftNeuralNet, self).__init__()
        self.hidden_units_dsl = hidden_units_dsl
        self.hidden_units_rem = hidden_units_rem
        self.layers_dsl = layers_dsl
        self.layers_rem = layers_rem
        self.dropout_rate = dropout_rate
        self.expected_conversion_rate_r = 0
        self.n_features = n_features
        self.dsl_penalty = dsl_penalty
        self.learning_rate = learning_rate

        # As we are not calling __init__() for the parent class, we need to
        # set the variables that the parent class would normally handle:
        self.model = self.set_model_architecture()
        self.ecrr = 0
        self.optimizer = OPTIMIZER(list(self.model['model_rem'].parameters()) +
                                   list(self.model['model_t'].parameters()) +
                                   list(self.model['model_c'].parameters()),
                                   lr=self.learning_rate)
        self.loss_fn = LOSS_FN()


    def set_model_architecture(self):
        """
        Method for setting the model architecture. Uses parameters
        stored in self.
        """
        # Create the DSL-layer for the control data:
        model_c = torch.nn.ModuleList()
        model_c.append(nn.Linear(self.n_features, self.hidden_units_dsl))
        model_c.append(nn.Sigmoid())
        model_c.append(nn.Dropout(self.dropout_rate))
        model_c.append(nn.BatchNorm1d(self.hidden_units_dsl))
        if self.layers_dsl > 1:
            for _ in range(self.layers_dsl - 1):
                model_c.append(nn.Linear(self.hidden_units_dsl, self.hidden_units_dsl))
                model_c.append(nn.Sigmoid())
                model_c.append(nn.Dropout(self.dropout_rate))
                model_c.append(nn.BatchNorm1d(self.hidden_units_dsl))
        # Create the DSL-layer for the treatment data:
        model_t = torch.nn.ModuleList()
        model_t.append(nn.Linear(self.n_features, self.hidden_units_dsl))
        model_t.append(nn.Sigmoid())
        model_t.append(nn.Dropout(self.dropout_rate))
        model_t.append(nn.BatchNorm1d(self.hidden_units_dsl))
        if self.layers_dsl > 1:
            for _ in range(self.layers_dsl - 1):
                model_t.append(nn.Linear(self.hidden_units_dsl, self.hidden_units_dsl))
                model_t.append(nn.Sigmoid())
                model_t.append(nn.Dropout(self.dropout_rate))
                model_t.append(nn.BatchNorm1d(self.hidden_units_dsl))

        # Create the connecting layer and the remaining layers:
        model_rem = torch.nn.ModuleList()
        model_rem.append(nn.Linear((self.hidden_units_dsl * 2), self.hidden_units_rem))
        model_rem.append(nn.Sigmoid())
        model_rem.append(nn.Dropout(self.dropout_rate))
        model_rem.append(nn.BatchNorm1d(self.hidden_units_rem))
        if self.layers_rem > 1:
            for _ in range(self.layers_rem - 1):
                model_rem.append(nn.Linear(self.hidden_units_rem, self.hidden_units_rem))
                model_rem.append(nn.Sigmoid())
                model_rem.append(nn.Dropout(self.dropout_rate))
                model_rem.append(nn.BatchNorm1d(self.hidden_units_rem))
        model_rem.append(nn.Linear(self.hidden_units_rem, 1))
        model_rem.append(nn.Sigmoid())

        model = {
            # Create Sequential models of ModuleLists:
            'model_c': torch.nn.Sequential(*model_c).to(DEVICE),
            'model_t': torch.nn.Sequential(*model_t).to(DEVICE),
            'model_rem': torch.nn.Sequential(*model_rem).to(DEVICE)
        }
        return model


    def set_training_mode(self, mode=False):
        """
        Args:
        mode (bool): True sets model into training mode (has effect
         on dropout layers), False into evaluation mode.
        """
        self.model['model_c'].train(mode)
        self.model['model_t'].train(mode)
        self.model['model_rem'].train(mode)

    def forward(self, x, t):
        """
        Args:
        x (torch.Tensor): features
        t (torch.Tensor): treatment label ('1' for treatment group,
         '0' for control group)
        """
        tmp_c = self.model['model_c'](x)
        tmp_t = self.model['model_t'](x) * t.view(-1, 1).repeat(1, self.hidden_units_dsl)
        res = self.model['model_rem'](torch.cat([tmp_c, tmp_t], dim=1))
        return res

    def estimate_loss(self, batch):
        """
        Args:
        batch (dict): A batch as returned by the DataLoader iterator.
        """
        self.set_training_mode(True)
        y_pred = self(batch['X'].to(DEVICE), batch['t'].to(DEVICE))
        loss = self.loss_fn(y_pred, batch['y'].view(-1, 1).to(DEVICE)) +\
               self.dsl_penalty * torch.norm(
                   self.model['model_rem'][0].weight[:, self.hidden_units_rem:] *
                   torch.mean(batch['t'].to(DEVICE).float()), 2)
        # This formulation equals putting penalty on [0].weigth... only for samples
        # where t=1.
        self.set_training_mode(False)
        return loss

    @classmethod
    def init_random_neural_net(cls,
                               n_features=12,
                               max_layers_dsl=8,
                               max_layers_rem=8,
                               max_hidden_units_dsl=256,
                               max_hidden_units_rem=256):
        """
        Method for initialization of neural net with randomized
        architecture and parameters. The set values (except for
        n_features) are parameters for random sampling, e.g.
        max_layers_dsl indicates that the number of layers for
        the dsl-layers is randomly drawn from {1, ... 8}.
        Note that this method returns a dict with a model _and_
        the model architecture in readable format.

        Args:
        n_features (int): Number of features in the data.
        max_layers_dsl (int): Number of maximum input layers in the
         dsl-parts of the network.
        max_layers_rem (int): Maximum number of layers in the remaining
         network.
        max_hidden_units_dsl (int): Maximum number of hidden units in a
         dsl-layer.
        max_hidden_units_rem (int): Maximum number of hidden units in the
         remaining network.
        """
        # The following assertions are here to ensure that network training
        # does not become too computationally complex.
        assert n_features > 0, "n_features needs to be larger than 0"
        assert 1 <= max_layers_dsl <= 16, "max_layers_dsl not within set limits."
        assert 1 <= max_layers_rem <= 16, "max_layers_rem not within set limits."
        assert 1 <= max_layers_dsl <= 512, "max_hidden_units_dsl not within set limits."
        assert 1 <= max_layers_rem <= 512, "max_hidden_units_rem not within set limits."

        random_state = random.getstate()  # Random state for later reproduction
        layers_dsl = random.randint(1, max_layers_dsl)
        layers_rem = random.randint(1, max_layers_rem)
        hidden_units_dsl = random.randint(1, max_hidden_units_dsl)
        hidden_units_rem = random.randint(1, max_hidden_units_rem)
        dropout_rate = random.uniform(0, 1)  # Float in [0, 1)
        penalty = random.uniform(0, 0.1)  # Float in [0, 0.1)
        learning_rate = random.uniform(0, 0.1)  # Default 0.01
        neural_net = cls(n_features,
                         layers_dsl, layers_rem,
                         hidden_units_dsl, hidden_units_rem,
                         dropout_rate,
                         dsl_penalty=penalty,
                         learning_rate=learning_rate)
        architecture = {'layers_dsl': layers_dsl,
                        'layers_rem': layers_rem,
                        'hidden_units_dsl': hidden_units_dsl,
                        'hidden_units_rem': hidden_units_rem,
                        'dropout_rate': dropout_rate,
                        'learning_rate': learning_rate,
                        'random_state': random_state}

        return {'neural_net': neural_net,
                'architecture': architecture}

    @classmethod
    def init_random_search_neural_net(cls,
                                      data,
                                      n_networks=3):
        """
        Method for training a large number of randomly sampled neural
        networks. The method returns the best model. This is a random
        search (i.e. not grid search).

        Args:
        data (load_data.DatasetCollection): Data containing all required
         sets.
        n_networks (int): Number of random network to sample and train.
        """
        if n_networks < 20:
            warnings.warn("n_networks is set to {}. ".format(n_networks) +
                          "It should preferably be > 100")
        # Prepare data for training. We are here using the alternative
        # sets present in the load_data.DatasetCollection object because
        # we need a validation set for early stopping and another one
        # for model selection:
        training_set = load_data.DatasetWrapper(data['training_set_2'])
        validation_set_2a = load_data.DatasetWrapper(data['validation_set_2a'])
        validation_set_2b = load_data.DatasetWrapper(data['validation_set_2b'])
        n_features = data['training_set_2']['X'].shape[1]
        best_model_ecrr = 0
        ecrr_list = []
        architecture_list = []
        for i in range(n_networks):
            # 1. Generate new network
            new_model = cls.init_random_neural_net(n_features)
            # 2. Train new network.
            new_model['neural_net'].fit(training_set, validation_set_2a)
            # 3. Compare to previous best network on "validation_data_2b".
            new_model_ecrr = new_model['neural_net'].get_metrics(validation_set_2b)
            # Store metrics on progress:
            ecrr_list.append(new_model_ecrr)
            architecture_list.append(new_model['architecture'])
            print("Model {} ecrr: {}\n".format(i, new_model_ecrr))
            if new_model_ecrr > best_model_ecrr:
                best_model_ecrr = new_model_ecrr
                best_model = new_model['neural_net']
        # 4. Return best network.
        # return {'best_model': best_model,
        #         'ecrr_list': ecrr_list,
        #         'architecture_list': architecture_list}
        return best_model

    @classmethod
    def init_grid_search_neural_net(cls, data, dsl_penalties=[.001]):
        """
        Method for finding best performing search among networks
        with different penalties. Returns the best model.

        Args:
        data (load_data.DatasetCollection): Data containing all required
         sets.
        """
        # Prepare data for training. We are here using the alternative
        # sets present in the load_data.DatasetCollection object because
        # we need a validation set for early stopping and another one
        # for model selection:
        training_set = load_data.DatasetWrapper(data['training_set_2'])
        validation_set_2a = load_data.DatasetWrapper(data['validation_set_2a'])
        validation_set_2b = load_data.DatasetWrapper(data['validation_set_2b'])
        n_features = data['training_set_2']['X'].shape[1]
        best_model_ecrr = 0
        ecrr_list = []
        for dsl_penalty in dsl_penalties:
            # 1. Generate new network
            new_model = cls(n_features=n_features,
                            dsl_penalty=dsl_penalty)
            # 2. Train new network.
            new_model.fit(training_set, validation_set_2a)
            # 3. Compare to previous best network on "validation_data_2b".
            new_model_ecrr = new_model.get_metrics(validation_set_2b)
            # Store metrics on progress:
            ecrr_list.append(new_model_ecrr)
            print("Model with dsl-penalty {} ecrr: {}\n".format(
                dsl_penalty, new_model_ecrr))
            if new_model_ecrr > best_model_ecrr:
                best_model_ecrr = new_model_ecrr
                best_model = new_model
        return best_model
