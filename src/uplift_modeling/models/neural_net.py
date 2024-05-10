"""
Basic neural net that can either take features as is as input
or embeddings (with small modifications to the code). This is
a form of sparse input which allows us to store all data in
it's sparse format on the GPU.
Note that this is a _response_ model for _classification_,
i.e. not a model predicting uplift nor a regression model!

Notes:
Perhaps add docstrings.
This classifier uses AUC-ROC as metric for best network (in contrast
to most dsl-networks that use AUUC (/ECRR)).
"""
import copy
import uplift_modeling.metrics.metrics as metrics  # Script with 'expected_calibration_error'
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader
import torch.nn as nn


if torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'
BATCH_SIZE = 200000
NUM_WORKERS = 16
# Default learning rate in pytorch (at least for Adagrad) is .01
LEARNING_RATE = .1

class NeuralNet(nn.Module):
    """
    A basic neural net class for binary classification.


    Args:
        differentiable_predictions  If True, predict(...) will be blocking gradients.
    """
    def __init__(self, n_features, n_hidden_units=128, dropout_rate=.5, dependent_var='y', 
                       differentiable_predictions=False):
        super().__init__()
        self.n_hidden_units = n_hidden_units
        self.dropout_rate = dropout_rate
        self.dependent_var = dependent_var
        self.max_auc_roc = 0
        self.max_ece = None
        self.model = torch.nn.Sequential(
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

        # Adamax does not support sparse input. Use Adagrad.
        self.optimizer = torch.optim.Adagrad(self.model.parameters(),
                                             lr=LEARNING_RATE)
        # optimizer = torch.optim.Adamax(model.parameters(), lr=LEARNING_RATE)

        # Both BCE and MSE would be acceptable loss-functions.
        # loss_fn = torch.nn.MSELoss()
        self.loss_fn = torch.nn.BCELoss()

        self._differentiable_predictions = differentiable_predictions

    def forward(self, x):
        return self.model(x)

    def fit(self, training_data, validation_data,
            n_epochs=10000,
            max_epochs_without_improvement=101,
            verbose=False):
        epochs_without_improvement = 0
        best_model = self.model
        # Create DataLoader outside of loop:
        tmp_training_dataloader = DataLoader(training_data, pin_memory=True, num_workers=NUM_WORKERS, batch_size=BATCH_SIZE)

        # Actual training loop:
        for i in range(n_epochs):
            if i % 50 == 0:
                # This is the slowest part of the code. Let's evaluate progress
                # only every 50th epoch:
                self.model.train(mode=False)
                validation_auc_roc, validation_ece = self.get_metrics(validation_data)
                if self.max_auc_roc < validation_auc_roc:
                    self.max_auc_roc = validation_auc_roc
                    # ECE is only a reasonable metric if the neural net is
                    # directly modeling uplift (e.g. not double-classifier)
                    self.max_ece = validation_ece
                    best_model = copy.deepcopy(self.model)
                    # Reset counter:
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 50
                    if epochs_without_improvement > max_epochs_without_improvement:
                        print("No improvement during last {} epochs after {} epochs.".format(max_epochs_without_improvement, i))
                        break
            self.model.train(mode=True)
            # Some data loader optimization:
            for batch in tmp_training_dataloader:
                #y_pred = self.model(batch['X'].to(DEVICE))
                #loss = self.loss_fn(y_pred, batch[self.dependent_var].view(-1, 1).to(DEVICE))
                # Maybe add tmp_next here to allow for data loading also during
                # training computation on GPU? Could speed up by 25-33%?
                # Two batches should fit in memory just fine.
                tmp_X = batch['X'].to(DEVICE, non_blocking=True)
                tmp_y = batch[self.dependent_var].to(DEVICE, non_blocking=True)
                y_pred = self.model(tmp_X)
                loss = self.loss_fn(y_pred, tmp_y.view(-1, 1))

                if verbose:
                    # Not tested:
                    print("Training set loss: {}".format(loss))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        if best_model is self.model:
            # Sanity check
            print("Model not updated. No improvement found in {} epochs.".format(i))
        # When done, set self.model to best model to date:
        self.model = best_model
        # Set to prediction mode:
        self.model.train(mode=False)
        if verbose:
            print("Max validation set AUC-ROC: {}".format(self.max_auc_roc))
            print("Max validation set ECE: {}".format(self.max_ece))
        return

    def predict_torch(self, dataset):
        # When predicting, model mode always needs to be set to false.
        self.model.train(mode=False)
        # Perhaps predict in batches?
        tmp_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
        predictions = None
        with torch.autograd.no_grad():
            for batch in tmp_dataloader:
                if predictions is None:
                    predictions = self.model(batch['X'].to(DEVICE)).cpu()
                else:
                    predictions = torch.cat((predictions,
                                             self.model(batch['X'].to(DEVICE)).cpu()))
        return predictions

    def predict(self, data):
        if self._differentiable_predictions: 
            return self.model(torch.from_numpy(data))

        self.model.train(mode=False)
        # Predict in one batch
        self.model.cpu()
        tmp = self.model(torch.from_numpy(data)).detach().numpy()
        self.model.to(DEVICE)
        return tmp

    def get_metrics(self, dataset):
        """
        Function for estimating auc-roc and ece for binary classifier.

        Args:
        dataset (load_data.DatasetWrapper): Dataset containing 'y' (true label).

        Notes:
        Estimating anything else than auc_roc is actually not that essential at
        this point as the model selection is done using only that metric, and
        the other metrics are later evaluated for the testing set.
        """
        # dataset as CriteoDataset
        predictions = self.predict_torch(dataset)
        auc_roc = roc_auc_score(dataset[:][self.dependent_var], predictions)
        ece = metrics.expected_calibration_error(dataset[:][self.dependent_var], predictions.numpy())
        #ece = metrics.expected_calibration_error(dataset[:]['y'], predictions)
        print("AUC-ROC: {}\t ECE: {}".format(auc_roc, ece))
        return((auc_roc, ece))
