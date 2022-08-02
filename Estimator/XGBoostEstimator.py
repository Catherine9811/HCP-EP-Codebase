from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from Estimator.Estimator import Estimator
from Estimator.PCAEstimator import PCAEstimator


class XGBoostEstimator(Estimator):

    def __init__(self, data_scaler=StandardScaler(),
                 model=XGBClassifier(
                        max_depth=2,
                        gamma=2,
                        eta=0.8,
                        reg_alpha=0.5,
                        reg_lambda=0.5
                    )):
        super().__init__(model, data_scaler=data_scaler)
