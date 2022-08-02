from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from Estimator.PCAEstimator import PCAEstimator


class GPCWithPCAEstimator(PCAEstimator):

    def __init__(self, data_scaler=StandardScaler(), model=GaussianProcessClassifier(1.0 * RBF(1.0)),
                 pca=PCA(n_components=8)):
        super().__init__(model, data_scaler=data_scaler, pca=pca)
