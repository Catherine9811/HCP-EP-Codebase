import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from DataAnalyzer.BaseAnalyzer import BaseAnalyzer


class CorrelationAnalyzer(BaseAnalyzer):
    def analyze(self):
        columns = []
        for column in self.DATA.columns:
            if column.startswith("Estimate"):
                columns.append(column)
        self.DATA = self.DATA.drop(columns=columns)
        self.DATA.columns = self.DATA.columns.str.rstrip("_volume")
        corr = self.DATA.astype(float).corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr, mask=mask, cmap=cmap, square=True, cbar_kws={"shrink": .5}, xticklabels=False)
        plt.tight_layout()
        plt.show()
