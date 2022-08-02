from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from DataAnalyzer.BaseAnalyzer import BaseAnalyzer
from DataAnalyzer.PerformanceAnalyzer import PerformanceAnalyzer
from DataLoader.DemographicDataLoader import DemographicDataLoader


class NestedCrossValidationModel(BaseEstimator):
    data_loader: DemographicDataLoader
    data_analyzer: PerformanceAnalyzer
    data_folder: BaseCrossValidator
    data_inner_folder: BaseCrossValidator
    data_estimator: BaseEstimator
    balance_data: bool

    def __init__(self, loader: DemographicDataLoader, estimator: BaseEstimator, analyzer: PerformanceAnalyzer,
                 n_splits: int = 5, balance_data: bool = False):
        self.data_loader = loader
        self.data_analyzer = analyzer
        self.data_estimator = estimator
        self.data_folder = StratifiedKFold(n_splits=n_splits)
        self.data_inner_folder = StratifiedKFold(n_splits=n_splits)
        self.balance_data = balance_data

    def process(self, param_grid: dict):
        feature, label = self.data_loader.get_features(), self.data_loader.get_labels()

        for train_index, test_index in self.data_folder.split(feature, label):
            X_train, X_test = feature[train_index], feature[test_index]
            y_train, y_test = label[train_index], label[test_index]
            search = GridSearchCV(estimator=self.data_estimator, param_grid=param_grid, cv=self.data_inner_folder)
            if self.balance_data:
                smote = SMOTE(sampling_strategy='auto', n_jobs=-1)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            search.fit(X_train, y_train)
            predictions = search.predict(X_test)
            print(f"Best Parameters: {search.best_params_}\n"
                  f"Best Score: {search.best_score_}")
            self.data_analyzer.record_metrics(y_test, predictions, params=search.best_params_, score=search.best_score_)
        print(f"Nested Grid Search Cross Validation "
              f"on {self.data_estimator.__class__.__name__} Complete on {param_grid}.")
        return self.data_analyzer.apply_metrics()
