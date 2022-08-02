import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from DataAnalyzer.BaseAnalyzer import BaseAnalyzer


class BoxAnalyzer(BaseAnalyzer):
    def analyze_cognitive(self):

        columns = []
        for column in self.DATA.columns:
            if not column.startswith("ER40") or column.startswith("ER40_"):
                columns.append(column)
        self.DATA = self.DATA.drop(columns=columns)
        self.DATA = self.DATA.assign(Groups=self.LABEL)
        self.DATA["Groups"] = self.DATA["Groups"].map({0: "Healthy Controls", 1: "Early Psychosis"})

        data = self.DATA.set_index('Groups').stack().reset_index().rename(columns={'level_1': 'Features', 0: 'Values'})

        sns.boxplot(x='Features', y='Values', data=data, hue='Groups')
        plt.show()


    def analyze(self):

        columns = []
        for column in self.DATA.columns:
            if column.startswith("Estimate"):
                columns.append(column)
        self.DATA = self.DATA.drop(columns=columns)
        self.DATA = self.DATA.assign(Groups=self.LABEL)
        self.DATA["Groups"] = self.DATA["Groups"].map({0: "Healthy Controls", 1: "Early Psychosis"})

        data = self.DATA.set_index('Groups').stack().reset_index().rename(columns={'level_1': 'Features', 0: 'Values'})

        sns.boxplot(x='Features', y='Values', data=data, hue='Groups')
        plt.show()
