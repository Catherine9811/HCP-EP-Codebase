from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from Estimator.PCAEstimator import PCAEstimator


class KNNWithPCAEstimator(PCAEstimator):

    def __init__(self, data_scaler=StandardScaler(), model=KNeighborsClassifier(),
                 pca=PCA(n_components=8)):
        super().__init__(model, data_scaler=data_scaler, pca=pca)
