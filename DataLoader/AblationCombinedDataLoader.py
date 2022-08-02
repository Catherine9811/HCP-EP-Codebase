from typing import List, Tuple

import pandas as pd
import numpy as np
import logging
import os

from sklearn.ensemble import IsolationForest

from DataLoader.BrainDataLoader import BrainDataLoader
from DataLoader.CognitiveDataLoader import CognitiveDataLoader
from DataLoader.CombinedDataLoader import CombinedDataLoader
from DataLoader.DemographicDataLoader import DemographicDataLoader


class AblationCombinedDataLoader(CombinedDataLoader):
    DROPPED = ""

    def __init__(self, clean_data: bool = False):
        super().__init__(clean_data=clean_data)

    def drop_feature(self, i):
        if 0 <= i < len(self.get_columns().columns):
            self.PROCESSED_DATA.drop(columns=[self.get_columns().columns[i]])
            self.DROPPED = self.get_columns().columns[i]
            print(f"Dropped {self.DROPPED}")
        return self
