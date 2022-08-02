import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
from DataAnalyzer.BaseAnalyzer import BaseAnalyzer


class FeatureAnalyzer(BaseAnalyzer):
    TOP_FEATURES: int = 6

    def analyze(self):
        forest = ExtraTreesClassifier(n_estimators=250, max_depth=5, random_state=1)
        forest.fit(self.DATA, self.LABEL)

        importance = forest.feature_importances_
        std = np.std(
            [tree.feature_importances_ for tree in forest.estimators_],
            axis=0
        )
        indices = np.argsort(importance)[::-1]
        indices = indices[:self.TOP_FEATURES]

        print('Top features:')
        features = []
        for f in range(self.TOP_FEATURES):
            print('%d. %s feature %d (%f)' % (f + 1, self.DATA.columns[indices[f]], indices[f], importance[indices[f]]))
            features.append(self.DATA.columns[indices[f]])
        plt.figure()
        plt.bar(
            range(self.TOP_FEATURES),
            importance[indices],
            yerr=std[indices],
        )
        plt.xticks(range(self.TOP_FEATURES), features)
        plt.tight_layout()
        plt.show()

        rfe = RFE(XGBClassifier(n_jobs=-1, random_state=1))

        rfe.fit(self.DATA, self.LABEL)

        print('Selected features:')
        print(rfe.support_)
