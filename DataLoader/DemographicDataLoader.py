from typing import List

import pandas as pd
import numpy as np
import os


class DemographicDataLoader:
    BASE_DATA_FILE: str = "hcp_psychosis_demographics.csv"
    STORAGE_PATH: str = "Data"
    BASE_DATA: pd.DataFrame
    EXTRA_DATA: pd.DataFrame
    PROCESSED_DATA: pd.DataFrame
    EXTRA_DATA_FILE: str
    CLEAN_DATA: bool

    def __init__(self, clean_data: bool = False):
        self.CLEAN_DATA = clean_data
        self.BASE_DATA = pd.read_csv(os.path.join(self.STORAGE_PATH, self.BASE_DATA_FILE))
        self.EXTRA_DATA = pd.read_csv(os.path.join(self.STORAGE_PATH, self.EXTRA_DATA_FILE))
        self.PROCESSED_DATA = self.process_data(self.BASE_DATA, self.EXTRA_DATA)
        self.PROCESSED_DATA = self.convert_categorical_data(self.PROCESSED_DATA)
        print(f"{len(self.PROCESSED_DATA)} entries loaded")

    def get_base_columns(self, exempt: str = "phenotype"):
        columns = self.BASE_DATA.columns.tolist()
        if exempt in columns:
            columns.remove(exempt)
        return columns

    def convert_categorical_data(self, df: pd.DataFrame, columns: List[str] = ["phenotype"]):
        for column in columns:
            print(df[column].astype("category").cat.categories)
            df[column] = df[column].astype("category").cat.codes
        return df

    def get_features(self):
        return self.get_processed_data().drop(columns=["phenotype"]).astype(float).to_numpy()

    def get_columns(self):
        return self.get_processed_data().drop(columns=["phenotype"]).astype(float)

    def get_labels(self):
        return self.get_processed_data()["phenotype"].astype('int').to_numpy()

    def process_data(self, base: pd.DataFrame, extra: pd.DataFrame):
        raise NotImplementedError

    def clean_data(self, data: pd.DataFrame):
        return data

    def get_processed_data(self):
        if self.CLEAN_DATA:
            return self.clean_data(self.PROCESSED_DATA.drop(columns=self.get_base_columns()))
        return self.PROCESSED_DATA.drop(columns=self.get_base_columns())

    def get_unprocessed_data(self):
        return self.PROCESSED_DATA
