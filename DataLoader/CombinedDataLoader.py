from typing import List, Tuple

import pandas as pd
import numpy as np
import logging
import os

from sklearn.ensemble import IsolationForest

from DataLoader.BrainDataLoader import BrainDataLoader
from DataLoader.CognitiveDataLoader import CognitiveDataLoader
from DataLoader.DemographicDataLoader import DemographicDataLoader


class CombinedDataLoader(DemographicDataLoader):
    EXTRA_DATA_LIST: List[DemographicDataLoader] = [
        BrainDataLoader(),
        CognitiveDataLoader()
    ]

    def __init__(self, clean_data: bool = False):
        self.CLEAN_DATA = clean_data
        self.BASE_DATA = pd.read_csv(os.path.join(self.STORAGE_PATH, self.BASE_DATA_FILE))
        self.PROCESSED_DATA = self.process_data(
            self.BASE_DATA,
            self.EXTRA_DATA_LIST
        )
        self.PROCESSED_DATA.to_csv("combined.csv")
        self.PROCESSED_DATA = self.convert_categorical_data(self.PROCESSED_DATA)
        print(f"{len(self.PROCESSED_DATA)} entries loaded")
        print(self.PROCESSED_DATA["phenotype"].value_counts())
        print(len(self.get_columns().columns))

    def process_data(self, base: pd.DataFrame, extra: List[DemographicDataLoader]):
        merged = None
        for loader in extra:
            if merged is None:
                merged = loader.get_unprocessed_data()
            else:
                merged = pd.merge(merged, loader.get_unprocessed_data(), how='inner',
                                  on=loader.get_base_columns(exempt=""))
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
        return data
