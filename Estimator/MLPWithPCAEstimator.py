from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from Estimator.PCAEstimator import PCAEstimator


class MLPWithPCAEstimator(PCAEstimator):

    def __init__(self, data_scaler=StandardScaler(), model=MLPClassifier(alpha=0.01, max_iter=5000),
                 pca=PCA(n_components=8)):
        super().__init__(model, data_scaler=data_scaler, pca=pca)
