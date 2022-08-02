import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from DataAnalyzer.BaseAnalyzer import BaseAnalyzer


class OutlierAnalyzer(BaseAnalyzer):
    def analyze(self):
        isf = IsolationForest(n_jobs=-1, random_state=1)
        isf.fit(self.DATA, self.LABEL)

        print(isf.score_samples(self.DATA))

        unique, count = np.unique(isf.predict(self.DATA), return_counts=True)
        counter = {kind: value for kind, value in zip(unique, count)}
        print(f"{counter[1]} Normal Points\n{counter[-1]} Outliers\n{counter[-1] * 100 / (counter[1] + counter[-1])}%")
