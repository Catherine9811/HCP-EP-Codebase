from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from Estimator.Estimator import Estimator
from Estimator.PCAEstimator import PCAEstimator


class LogisticEstimator(Estimator):

    def __init__(self, data_scaler=StandardScaler(),
                 model=LogisticRegression(penalty='l2', C=0.1)):
        super().__init__(model, data_scaler=data_scaler)
