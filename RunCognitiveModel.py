from DataAnalyzer.PerformanceAnalyzer import PerformanceAnalyzer
from DataLoader.CognitiveDataLoader import CognitiveDataLoader
from Estimator.GPCWithPCAEstimator import GPCWithPCAEstimator
from Estimator.KNNWithPCAEstimator import KNNWithPCAEstimator
from Estimator.LogisticEstimator import LogisticEstimator
from Estimator.TreeWithPCAEstimator import TreeWithPCAEstimator
from Estimator.XGBoostEstimator import XGBoostEstimator
from Model.NestedCrossValidationModel import NestedCrossValidationModel
from Estimator.MLPWithPCAEstimator import MLPWithPCAEstimator
from Estimator.SVMWithPCAEstimator import SVMWithPCAEstimator
import numpy as np


def run_mlp():
    model = NestedCrossValidationModel(loader=CognitiveDataLoader(), estimator=MLPWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    # model.process({"pca__n_components": range(4, 16), "model__alpha": np.linspace(0.01, 1, 5)})
    model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})


"""
Nested Grid Search Cross Validation on MLPWithPCAEstimator Complete on {'pca__n_components': [11, 12], 'model__alpha': [0.25, 0.5]}.

Weighted F1-score: 0.7340525388548936 (0.06957226995382054)
Balanced Accuracy: 0.674949494949495 (0.061457487361429995)
Accuracy: 0.7480031257117512 (0.07910618011244885)
Sensitivity: 0.7409274193548387 (0.07520458568888494)
Specificity: 0.5044444444444445 (0.10076252487781895)
"""


def run_svm():
    model = NestedCrossValidationModel(loader=CognitiveDataLoader(), estimator=SVMWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 16), "model__C": np.linspace(0.01, 1, 5)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})

"""
Nested Grid Search Cross Validation on SVMWithPCAEstimator Complete on {'pca__n_components': range(4, 16), 'model__C': array([0.01  , 0.2575, 0.505 , 0.7525, 1.    ])}.

Weighted F1-score: 0.717847705781387 (0.03009786134190873)
Balanced Accuracy: 0.6572727272727273 (0.050189519407758794)
Accuracy: 0.7492413033622792 (0.05600484461323666)
Sensitivity: 0.7342741935483872 (0.03697055139073902)
Specificity: 0.4600000000000001 (0.1823847513660728)
"""


def run_gpc():
    model = NestedCrossValidationModel(loader=CognitiveDataLoader(), estimator=GPCWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 16)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})


"""
Nested Grid Search Cross Validation on GPCWithPCAEstimator Complete on {'pca__n_components': range(4, 16)}.

Weighted F1-score: 0.7169773546707011 (0.030341043089967507)
Balanced Accuracy: 0.6497979797979798 (0.05050565656202026)
Accuracy: 0.7495396430572592 (0.04433015466870655)
Sensitivity: 0.740725806451613 (0.020849152856120982)
Specificity: 0.4177777777777778 (0.1795192482544063)
"""


def run_knn():
    model = NestedCrossValidationModel(loader=CognitiveDataLoader(), estimator=KNNWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 16), "model__n_neighbors": range(1, 16)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})

"""
Nested Grid Search Cross Validation on KNNWithPCAEstimator Complete on {'pca__n_components': range(4, 16), 'model__n_neighbors': range(1, 16)}.

Weighted F1-score: 0.6929066123161496 (0.08541225590801305)
Balanced Accuracy: 0.6713131313131313 (0.07720595253096521)
Accuracy: 0.7433123028915686 (0.08855754492894415)
Sensitivity: 0.6901209677419355 (0.09689508577984252)
Specificity: 0.6244444444444444 (0.18045210711699372)
"""


def run_tree():
    model = NestedCrossValidationModel(loader=CognitiveDataLoader(), estimator=TreeWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 16)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})

"""
Nested Grid Search Cross Validation on TreeWithPCAEstimator Complete on {'pca__n_components': range(4, 16)}.

Weighted F1-score: 0.6800402080609842 (0.0675848361742293)
Balanced Accuracy: 0.6209090909090909 (0.08170249958804689)
Accuracy: 0.6821992323023548 (0.07176493092173114)
Sensitivity: 0.6838709677419356 (0.06693609113599387)
Specificity: 0.4600000000000001 (0.14666666666666667)
"""


def run_xgboost():
    model = NestedCrossValidationModel(loader=CognitiveDataLoader(clean_data=True), estimator=XGBoostEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({})
"""
Nested Grid Search Cross Validation on XGBoostEstimator Complete on {}.

Weighted F1-score: 0.6927371962443537 (0.0667884878127087)
Balanced Accuracy: 0.6236363636363637 (0.05943343701456055)
Accuracy: 0.7045256795114698 (0.07815074679272611)
Sensitivity: 0.7028225806451612 (0.07541885727229009)
Specificity: 0.42000000000000004 (0.082671445501059)
"""



def run_logistic():
    model = NestedCrossValidationModel(loader=CognitiveDataLoader(clean_data=True), estimator=LogisticEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({})

"""
Nested Grid Search Cross Validation on LogisticEstimator Complete on {}.

Weighted F1-score: 0.7363170052233766 (0.04575967510277415)
Balanced Accuracy: 0.6634343434343435 (0.06170105097863548)
Accuracy: 0.7660478905132686 (0.054286262957783465)
Sensitivity: 0.7598790322580646 (0.04041521617201878)
Specificity: 0.4177777777777778 (0.15565076451411075)
"""


if __name__ == '__main__':
    run_gpc()