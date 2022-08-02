from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF


class PCAEstimator(BaseEstimator):

    def __init__(self, model, data_scaler=StandardScaler(),
                 pca=PCA(n_components=8)):
        self.data_scaler = data_scaler
        self.model = model
        self.pca = pca

    def fit(self, X, y):
        self.data_scaler.fit(X)
        X = self.data_scaler.transform(X)
        self.pca.fit(X)
        pca_X = self.pca.transform(X)
        self.model.fit(pca_X, y)
        return self

    def predict(self, X):
        X = self.data_scaler.transform(X)
        pca_X = self.pca.transform(X)
        predictions = self.model.predict(pca_X)
        return predictions

    def score(self, X, y_true):
        return balanced_accuracy_score(y_true, self.predict(X))
