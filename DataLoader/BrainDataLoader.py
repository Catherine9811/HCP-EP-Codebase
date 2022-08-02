import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
import logging
import os

from DataLoader.DemographicDataLoader import DemographicDataLoader


class BrainDataLoader(DemographicDataLoader):
    EXTRA_DATA_FILE: str = "hcp_psychosis_freesurferData.csv"

    def __init__(self, clean_data: bool = False):
        super().__init__(clean_data=clean_data)

    def process_data(self, base: pd.DataFrame, extra: pd.DataFrame):
        extra = extra.assign(src_subject_id=lambda row: (
                    row.image_id.str.split('_').str[0].str.replace("sub-", "").astype(int) / 100).astype(int))
        merged = pd.merge(extra, base, how='inner', left_on="src_subject_id", right_on="src_subject_id")
        # Drop brain data
        merged = merged.drop(columns=["image_id"])
        # Process brain data
        columns = extra.columns.tolist()
        columns.remove("EstimatedTotalIntraCranialVol")
        columns.remove("image_id")
        columns.remove("src_subject_id")
        merged[columns] = merged[columns].div(merged["EstimatedTotalIntraCranialVol"], axis=0)
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
            columns = [True, True, False, True, True, True, False, True, True, True, True, True, True, True, True, False, False, False, True, False, False, False, False, True, False, False, False, False, True, False, True, False, False, False, False, False, False, False, False, True, True, True, False, True, False, False, False, False, True, False, False, True, True, True, True, True, False, False, True, True, True, True, True, False, True, True, False, True, True, True, False, True, True, True, True, True, True, False, True, True, False, False, True, True, False, False, True, True, False, False, True, False, False, False, True, False, False, False, False, False, False, False]
            columns.insert(index, True)
            columns_to_delete = np.where([not value for value in columns])[0]
            data = data.drop(index=columns_to_delete, axis=1)
            print(f"Dropped {len(columns_to_delete)} features in {self.__class__.__name__}")
        return data
