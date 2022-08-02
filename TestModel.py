import unittest

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from DataAnalyzer.PerformanceAnalyzer import PerformanceAnalyzer
from DataLoader.AblationCombinedDataLoader import AblationCombinedDataLoader
from DataLoader.BrainDataLoader import BrainDataLoader
from DataLoader.CognitiveDataLoader import CognitiveDataLoader
from DataLoader.CombinedDataLoader import CombinedDataLoader
from Estimator.GPCWithPCAEstimator import GPCWithPCAEstimator
from Estimator.KNNWithPCAEstimator import KNNWithPCAEstimator
from Estimator.SVMWithPCAEstimator import SVMWithPCAEstimator
from Estimator.TreeWithPCAEstimator import TreeWithPCAEstimator
from Model.EnsembleModel import EnsembleModel
from Model.NestedCrossValidationModel import NestedCrossValidationModel
from Estimator.MLPWithPCAEstimator import MLPWithPCAEstimator


class GridSearchTestCase(unittest.TestCase):
    def test_mlp(self):
        model = NestedCrossValidationModel(loader=CognitiveDataLoader(), estimator=MLPWithPCAEstimator(),
                                           analyzer=PerformanceAnalyzer())
        model.process({"pca__n_components": range(4, 22), "model__alpha": np.linspace(0.01, 1, 100)})

    def test_clean_combined_ensemble(self):
        model = EnsembleModel(
            loader=CombinedDataLoader(clean_data=True), estimator_list=[
                MLPWithPCAEstimator(model=MLPClassifier(alpha=0.25, max_iter=5000), pca=PCA(n_components=16)),
                KNNWithPCAEstimator(model=KNeighborsClassifier(n_neighbors=12), pca=PCA(n_components=16)),
                GPCWithPCAEstimator(pca=PCA(n_components=16)),
                SVMWithPCAEstimator(model=SVC(kernel="rbf"), pca=PCA(n_components=16)),
                TreeWithPCAEstimator(pca=PCA(n_components=16)),
            ],
            analyzer=PerformanceAnalyzer(),
            criteria=0.6
        )
        model.process()

        """
Weighted F1-score: 0.7127745779464278 (0.07693962206499325)
Balanced Accuracy: 0.6747619047619047 (0.05124403498153922)
Accuracy: 0.7606379928315412 (0.05650888555150001)
Sensitivity: 0.723010752688172 (0.08743092676917834)
Specificity: 0.54 (0.17321933573248957)
        """

    def test_combined_ensemble(self):
        model = EnsembleModel(
            loader=CombinedDataLoader(clean_data=False), estimator_list=[
                MLPWithPCAEstimator(model=MLPClassifier(alpha=0.25, max_iter=5000), pca=PCA(n_components=16)),
                KNNWithPCAEstimator(model=KNeighborsClassifier(n_neighbors=12), pca=PCA(n_components=16)),
                GPCWithPCAEstimator(pca=PCA(n_components=16)),
                SVMWithPCAEstimator(model=SVC(kernel="rbf"), pca=PCA(n_components=16)),
                TreeWithPCAEstimator(pca=PCA(n_components=16)),
            ],
            analyzer=PerformanceAnalyzer(),
            criteria=0.6
        )
        model.process()
        """
Weighted F1-score: 0.7647161871947507 (0.12590054439554474)
Balanced Accuracy: 0.7408658008658009 (0.13060724436886686)
Accuracy: 0.7935738003620192 (0.10348945516592785)
Sensitivity: 0.7677419354838709 (0.129675814466076)
Specificity: 0.68 (0.22070593809662464)

Weighted F1-score: 0.7706835891763267 (0.1315527851964181)
Balanced Accuracy: 0.7467388167388167 (0.14124268399600812)
Accuracy: 0.7965821991628443 (0.11242792955773494)
Sensitivity: 0.7741935483870968 (0.13378349260211433)
Specificity: 0.6822222222222223 (0.2361836489882031)
        """

    def test_cognitive_ensemble(self):
        model = EnsembleModel(
            loader=CognitiveDataLoader(), estimator_list=[
                MLPWithPCAEstimator(model=MLPClassifier(alpha=0.01, max_iter=5000), pca=PCA(n_components=12)),
                KNNWithPCAEstimator(model=KNeighborsClassifier(n_neighbors=12), pca=PCA(n_components=12)),
                GPCWithPCAEstimator(pca=PCA(n_components=12)),
                SVMWithPCAEstimator(model=SVC(kernel="rbf"), pca=PCA(n_components=12)),
                TreeWithPCAEstimator(pca=PCA(n_components=12)),
            ],
            analyzer=PerformanceAnalyzer(),
            criteria=0.6
        )
        model.process()
        """
Weighted F1-score: 0.7432498033205263 (0.04710740682086552)
Balanced Accuracy: 0.6818181818181819 (0.052521367377680706)
Accuracy: 0.7612066618903925 (0.06469195694664186)
Sensitivity: 0.7534274193548388 (0.05279256103392115)
Specificity: 0.5 (0.12292725943057183)
        """

    def test_brain_ensemble(self):
        model = EnsembleModel(
            loader=BrainDataLoader(), estimator_list=[
                MLPWithPCAEstimator(model=MLPClassifier(alpha=0.01, max_iter=5000), pca=PCA(n_components=8)),
                KNNWithPCAEstimator(model=KNeighborsClassifier(n_neighbors=12), pca=PCA(n_components=8)),
                GPCWithPCAEstimator(pca=PCA(n_components=8)),
                SVMWithPCAEstimator(model=SVC(kernel="rbf"), pca=PCA(n_components=8)),
                TreeWithPCAEstimator(pca=PCA(n_components=8)),
            ],
            analyzer=PerformanceAnalyzer(),
            criteria=0.6
        )
        model.process()
        """
Weighted F1-score: 0.6787895093005074 (0.07261579921369332)
Balanced Accuracy: 0.6046666666666667 (0.07487417667270246)
Accuracy: 0.7020862549354525 (0.08495536623591211)
Sensitivity: 0.7133333333333334 (0.06653691076257721)
Specificity: 0.3 (0.11026306302218335)
        """

        """
Cognitive Data
Weighted F1-score: 0.7432498033205263 (0.04710740682086552)
Balanced Accuracy: 0.6818181818181819 (0.052521367377680706)
Accuracy: 0.7612066618903925 (0.06469195694664186)
Sensitivity: 0.7534274193548388 (0.05279256103392115)
Specificity: 0.5 (0.12292725943057183)
"""

        """
Brain Data
Weighted F1-score: 0.6919273547943448 (0.06088509702848235)
Balanced Accuracy: 0.6173333333333334 (0.06495900734652095)
Accuracy: 0.7272997374026295 (0.09037802651264248)
Sensitivity: 0.7246031746031746 (0.05849157890001932)
Specificity: 0.31666666666666665 (0.09393939393939393)
"""
    def test_ablation_combined_ensemble(self):
        for i in range(len(AblationCombinedDataLoader(clean_data=False).get_columns().columns)):
            for j in range(10):
                print(f"Run {j}")
                model = EnsembleModel(
                    loader=AblationCombinedDataLoader(clean_data=False).drop_feature(i), estimator_list=[
                        MLPWithPCAEstimator(model=MLPClassifier(alpha=0.25, max_iter=5000), pca=PCA(n_components=16)),
                        KNNWithPCAEstimator(model=KNeighborsClassifier(n_neighbors=12), pca=PCA(n_components=16)),
                        GPCWithPCAEstimator(pca=PCA(n_components=16)),
                        SVMWithPCAEstimator(model=SVC(kernel="rbf"), pca=PCA(n_components=16)),
                        TreeWithPCAEstimator(pca=PCA(n_components=16)),
                    ],
                    analyzer=PerformanceAnalyzer(),
                    criteria=0.6
                )
                model.process()


if __name__ == '__main__':
    unittest.main()
