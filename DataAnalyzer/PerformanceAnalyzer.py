import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from DataAnalyzer.BaseAnalyzer import BaseAnalyzer


class PerformanceAnalyzer(BaseAnalyzer):
    metrics = []

    def __init__(self):
        super().__init__(pd.DataFrame(), pd.DataFrame())

    def analyze(self):
        balanced_acc_scores = [x['balanced-acc'] for x in self.metrics]
        return np.mean(balanced_acc_scores)

    def reset_metrics(self):
        self.metrics = []

    def record_metrics(self, reference, prediction, **kwargs):
        report = classification_report(reference, prediction, output_dict=True)
        tn, fp, fn, tp = confusion_matrix(reference, prediction).ravel()
        specificity = tn / (tn + fp)
        report["weighted avg"]["specificity"] = specificity
        report["weighted avg"]["balanced-acc"] = balanced_accuracy_score(reference, prediction)
        report["weighted avg"]["parameters"] = kwargs
        self.metrics.append(report["weighted avg"])

    def apply_metrics(self, info: str = ""):
        f1_scores = [x['f1-score'] for x in self.metrics]
        sensitivity_scores = [x['recall'] for x in self.metrics]
        accuracy_scores = [x['precision'] for x in self.metrics]
        specificity_scores = [x['specificity'] for x in self.metrics]
        balanced_acc_scores = [x['balanced-acc'] for x in self.metrics]
        print(
            f"{info}\n"
            f"Weighted F1-score: {np.mean(f1_scores)} ({np.std(f1_scores)})\n"
            f"Balanced Accuracy: {np.mean(balanced_acc_scores)} ({np.std(balanced_acc_scores)})\n"
            f"Accuracy: {np.mean(accuracy_scores)} ({np.std(accuracy_scores)})\n"
            f"Sensitivity: {np.mean(sensitivity_scores)} ({np.std(sensitivity_scores)})\n"
            f"Specificity: {np.mean(specificity_scores)} ({np.std(specificity_scores)})")
        for item in self.metrics:
            if "parameters" in item and item["parameters"] is not None and len(item["parameters"]) > 0:
                print(f"Extra Parameters: {item['parameters']}")
        return np.mean(balanced_acc_scores)
