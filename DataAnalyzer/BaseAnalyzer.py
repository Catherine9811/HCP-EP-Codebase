import pandas as pd


class BaseAnalyzer:
    DATA: pd.DataFrame
    LABEL: pd.DataFrame

    def __init__(self, df: pd.DataFrame, label: pd.DataFrame):
        self.DATA = df
        self.LABEL = label

    def analyze(self):
        pass
