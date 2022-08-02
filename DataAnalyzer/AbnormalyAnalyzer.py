import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from DataAnalyzer.BaseAnalyzer import BaseAnalyzer


class AbnormalyAnalyzer(BaseAnalyzer):
    def analyze(self):
        n = len(self.DATA.columns)
        c = 2
        for index, column in enumerate(self.DATA.columns):
            plt.subplot(c, n // c, index + 1)
            plt.hist(self.DATA[column], bins=32)
            plt.xlabel(column)
        plt.show()
