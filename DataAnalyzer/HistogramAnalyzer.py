import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import normaltest
from DataAnalyzer.BaseAnalyzer import BaseAnalyzer


class HistogramAnalyzer(BaseAnalyzer):

    def analyze(self):

        plt.figure(figsize=(15, 10), dpi=300)
        for column in self.DATA.columns:
            if column.startswith("Estimate"):
                self.DATA[column] /= 1e6
        sns.set_theme(style="darkgrid")
        num = len(self.DATA.columns)
        col = 6
        row = num // col + 1
        for index, column in enumerate(self.DATA.columns):
            plt.subplot(row, col, index + 1)
            _, pvalue = normaltest(self.DATA[column])
            plt.title(f"p = {pvalue.round(3)} (two-tailed)")
            ax = sns.distplot(self.DATA[column])
            ax.set(ylabel=None)
            print(f"{column} {pvalue}")
        plt.tight_layout()
        plt.savefig("brain.png")
        # plt.show()
