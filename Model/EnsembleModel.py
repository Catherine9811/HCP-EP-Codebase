from typing import List

from sklearn.base import BaseEstimator

from DataAnalyzer.PerformanceAnalyzer import PerformanceAnalyzer
from DataLoader.DemographicDataLoader import DemographicDataLoader
from Model.CrossValidationModel import CrossValidationModel


class EnsembleModel(CrossValidationModel):
    data_estimator_list: List[BaseEstimator]
    data_criteria: float

    def __init__(self, loader: DemographicDataLoader, estimator_list: List[BaseEstimator],
                 analyzer: PerformanceAnalyzer, n_splits: int = 5, criteria: float = 0.5):
        self.data_estimator_list = estimator_list
        self.data_criteria = criteria
        super().__init__(loader=loader, estimator=BaseEstimator(), analyzer=analyzer, n_splits=n_splits)

    def process(self):
        feature, label = self.data_loader.get_features(), self.data_loader.get_labels()
        for train_index, test_index in self.data_folder.split(feature, label):
            X_train, X_test = feature[train_index], feature[test_index]
            y_train, y_test = label[train_index], label[test_index]
            predictions = 0.0
            for data_estimator in self.data_estimator_list:
                data_estimator.fit(X_train, y_train)
                prediction = data_estimator.predict(X_test)
                predictions += prediction
            predictions /= len(self.data_estimator_list)
            predictions[predictions >= self.data_criteria] = 1
            predictions[predictions < self.data_criteria] = 0
            self.data_analyzer.record_metrics(y_test, predictions)
        return self.data_analyzer.apply_metrics()