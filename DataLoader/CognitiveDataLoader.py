import pandas as pd
import numpy as np
import logging
import os

from sklearn.ensemble import IsolationForest

from DataLoader.DemographicDataLoader import DemographicDataLoader


class CognitiveDataLoader(DemographicDataLoader):
    EXTRA_DATA_FILE: str = "cognition.csv"

    def __init__(self, clean_data: bool = False):
        super().__init__(clean_data=clean_data)

    def process_data(self, base: pd.DataFrame, extra: pd.DataFrame):
        extra = extra.dropna(axis=1)
        merged = pd.merge(extra, base, how='inner', left_on="id", right_on="subjectkey")
        # Drop brain data
        merged = merged.drop(columns=["id"])

        return merged

    def clean_data(self, data: pd.DataFrame):
        if True:
            # Outlier Detection
            isf = IsolationForest(n_jobs=-1, random_state=1)
            isf.fit(data.drop(columns=["phenotype"]), data["phenotype"])
            predictions = isf.predict(data.drop(columns=["phenotype"]))
            rows_to_delete = np.where(predictions == -1)[0]
            data = data.drop(index=rows_to_delete, axis=0)
            print(f"Dropped {len(rows_to_delete)} items in {self.__class__.__name__}")
        if False:
            # Feature Extraction
            index = data.columns.get_loc("phenotype")
            columns = [True, False, True, False, False, False, True, False, True, True, False, False, False, True, True, True]
            columns.insert(index, True)
            columns_to_delete = np.where([not value for value in columns])[0]
            data = data.drop(index=columns_to_delete, axis=1)
            print(f"Dropped {len(columns_to_delete)} features in {self.__class__.__name__}")
        return data
