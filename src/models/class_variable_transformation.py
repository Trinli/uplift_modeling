"""
Class-variable transformation following Jaskowski & Jaroscewicz (2012)
for uplift modeling. Note that there is also another version by Athey &
Imbens (2014?) not currently implemented here.

Quick theory recap:
Transformation
z = True if (y == True and t == True)
z = True if (y == False and t == False)
z = False else.
This leads to:
p(y | do(t=1)) - p(y | do(t=0)) = 2 * p(z) - 1
"""
from numpy import float64, where  #, reshape
from src.data.load_data import DatasetWrapper
import warnings


# Number of epochs for neural network training:
N_EPOCHS = 1000
if N_EPOCHS < 100:
    warnings.warn("Number of epochs for neural net training " +
                  "is set to {}. It should preferably ".format(N_EPOCHS) +
                  "exceed 1000.")

class ClassVarTransform():
    """
    Base-class for CVT-Z -based approaches.
    Modify the base-learners to suit this.
    """
    def __init__(self, base_learner):
        """
        ?
        """
        self.model = base_learner()

    def fit(self, data, undersampling=None):
        # This should model 'z', not 'y'
        # We already have 'z' stored in teh dataset
        tmp = data['training_set', undersampling, 'all']
        self.model.fit(tmp['X'], tmp['z'])

    def predict_uplift(self, X):
        """
        Function for estimating uplift with class-variable
        transformation.

        Args:
        data (np.array([[float]])): Array with features ('X').
        """
        predictions = self.model.predict(X)
        uplift_predictions = 2.0 * predictions - 1.0
        return uplift_predictions


class LogisticRegression(ClassVarTransform):

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        super().__init__(LogisticRegression)

    def predict_uplift(self, X):
        """
        Logistic regression from sklearn predicts probabilities
        for both classes. Hence we need to overwrite this method
        in the parent class.
        """
        tmp = self.model.predict_proba(X)
        # The logistic regression library predicts probabilities for
        # both classes. We want the probability for the true class
        # only:
        true_idx = where(self.model.classes_ == 1.0)[0][0]
        return 2.0 * tmp[:, true_idx].astype(float64) - 1.0


class LogisticReWeight(LogisticRegression):
    """
    This is the version of the CVT that Diemert & al. used in their paper
    (2018). The paper does not say anything about reweighting, this is
    something that Diemert mentioned in a linkedin-conversation.
    Instead of satisfying the requirement by the class-variable transformation
    that p(t) == p(c) (== 1/2) by resampling, we simply reweight the
    importance of samples in training to achieve the same.

    Note:
    Even with reweighting, it seems unlikely that the class-variable
    transformation will produce anything useful in cases where the
    class-imbalance (positive to negative) is large. E.g. the case
    for the Criteo-uplift dataset is one where the positive rate
    is around 0.2%. In such a case, 99.8% of the samples are samples where
    y_i == 0. The class-variable transformation just labels 15% of those
    as z = 1 and leaves the rest as z = 0.
    """
    def fit(self, data):
        """
        Fitting mechanism for class variable transformation partly following
        Jaskowski & Jaroszewicz, but using reweighting of samples rather than
        resampling to fulfil the requirements of the class-variable
        transformation (i.e. p(t) == p(c)).

        Args:
        data (load_data.DataCollection): Data for uplift modeling.

        Note:
        Due to penalty in the logistic regression module, the results
        actually differ depending on whether we set treatment or control
        sample weight as 1 (and adjust the other).
        """
        N_t = sum(data['training_set']['t'])
        N_c = sum(~data['training_set']['t'])
        assert (N_c + N_t) == len(data['training_set']['t']), "Error in calculating " +\
            "treatment and control samples."
        # Set weight for control samples (negation of treatment):
        weight_vec = (N_t / N_c) * ~data['training_set']['t']
        # The weight could equally well be set for treatment samples:
        # weight_vec = (N_c / N_t) * data['training_set']['t']
        # Set weight for treatment samples to 1:
        weight_vec[weight_vec == 0] = 1
        self.model.fit(X=data['training_set']['X'],
                       y=data['training_set']['z'],
                       sample_weight=weight_vec)


class CVTNeuralNet(ClassVarTransform):
    """
    Class for neural net with class-variable transformation.
    """

    def __init__(self, n_features):
        """
        Args:
        n_features (int): Needed for initialization of neural network.
        """
        from models.neural_net import NeuralNet
        self.n_features = n_features
        self.model = NeuralNet(self.n_features, dependent_var='z')


    def fit(self, data, undersampling=None):  # Undersampling not implemented
        # for training_set_2 and validation_sets
        """
        Method for fitting neural network.

        Args:
        data (load_data.DatasetCollection): Contains both training set and
        validation set for early stopping. Also contains a second validation
        set for model selection.
        undersampling (str): {None, '11', '1111'}. Only training data is undersampled.
         Validation set is not as we want best performance on data with natural
         sampling rate.
        hyperparameters (?): ???
        """
        training_data = DatasetWrapper(data['training_set_2', undersampling,
                                            'all'])
        validation_data = DatasetWrapper(data['validation_set_2a', None,
                                              'all'])

        # Split between 'X' and 'z' needs to be done _after_ batching, i.e.
        # in training loop for neural net. Perhaps pass 'dependent var'?
        self.model.fit(training_data, validation_data, n_epochs=N_EPOCHS)

    def predict_uplift(self, X):
        """
        Wrapper for predict_uplift from parent class.
        The output needs reshaping.
        """
        predicted_uplift = super().predict_uplift(X)
        predicted_uplift = predicted_uplift.reshape(-1).astype(float64)
        return predicted_uplift


class CVTRandomForest(ClassVarTransform):
    """
    Class-variable transformation with random forest.
    """
    def __init__(self):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier()

    def fit(self, data):
        """
        Method for fitting the uplift model.

        Args:
        data (load_data.DatasetCollection): The data
        """
        self.model.fit(X=data['training_set']['X'], y=data['training_set']['z'])

    def predict_uplift(self, X):
        """
        Method for predicting uplift.

        Args:
        X (numpy.array): The features
        """
        pred = self.model.predict_proba(X)
        # Find the "true" class:
        idx = where(self.model.classes_ == 1.0)[0][0]
        # Make the last step of the transformation:
        pred = 2 * pred[:, idx] - 1
        return pred
