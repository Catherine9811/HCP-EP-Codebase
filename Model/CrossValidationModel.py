from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from DataAnalyzer.BaseAnalyzer import BaseAnalyzer
from DataAnalyzer.PerformanceAnalyzer import PerformanceAnalyzer
from DataLoader.DemographicDataLoader import DemographicDataLoader


class CrossValidationModel(BaseEstimator):
    data_loader: DemographicDataLoader
    data_analyzer: PerformanceAnalyzer
    data_folder: BaseCrossValidator
    data_estimator: BaseEstimator

    def __init__(self, loader: DemographicDataLoader, estimator: BaseEstimator, analyzer: PerformanceAnalyzer,
                 n_splits: int = 5):
        self.data_loader = loader
        self.data_analyzer = analyzer
        self.data_estimator = estimator
        self.data_folder = StratifiedKFold(n_splits=n_splits)

    def process(self):
        feature, label = self.data_loader.get_features(), self.data_loader.get_labels()
        for train_index, test_index in self.data_folder.split(feature, label):
            X_train, X_test = feature[train_index], feature[test_index]
            y_train, y_test = label[train_index], label[test_index]
            self.data_estimator.fit(X_train, y_train)
            predictions = self.data_estimator.predict(X_test)
            self.data_analyzer.record_metrics(y_test, predictions)
        return self.data_analyzer.apply_metrics()

    def summary(self):
        self.data_analyzer.apply_metrics(self.__class__.__name__)
