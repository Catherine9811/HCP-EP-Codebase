from DataAnalyzer.PerformanceAnalyzer import PerformanceAnalyzer
from DataLoader.BrainDataLoader import BrainDataLoader
from DataLoader.CognitiveDataLoader import CognitiveDataLoader
from Estimator.GPCWithPCAEstimator import GPCWithPCAEstimator
from Estimator.KNNWithPCAEstimator import KNNWithPCAEstimator
from Estimator.LogisticEstimator import LogisticEstimator
from Estimator.SVMEstimator import SVMEstimator
from Estimator.TreeWithPCAEstimator import TreeWithPCAEstimator
from Estimator.XGBoostEstimator import XGBoostEstimator
from Model.NestedCrossValidationModel import NestedCrossValidationModel
from Estimator.MLPWithPCAEstimator import MLPWithPCAEstimator
from Estimator.SVMWithPCAEstimator import SVMWithPCAEstimator
import numpy as np


def run_mlp():
    model = NestedCrossValidationModel(loader=BrainDataLoader(), estimator=MLPWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    # model.process({"pca__n_components": range(4, 16), "model__alpha": np.linspace(0.01, 1, 5)})
    model.process({"pca__n_components": range(4, 20), "model__alpha": [0.01, 0.25, 0.5, 1.0]})


"""
Nested Grid Search Cross Validation on MLPWithPCAEstimator Complete on {'pca__n_components': range(4, 20), 'model__alpha': [0.01, 0.25, 0.5, 1.0]}.

Weighted F1-score: 0.6540801200946129 (0.05878950459623723)
Balanced Accuracy: 0.5898484848484848 (0.06130440463415505)
Accuracy: 0.6702451120308263 (0.06507677041443921)
Sensitivity: 0.6733333333333332 (0.06993788411272377)
Specificity: 0.353030303030303 (0.10449054333070706)
"""

def run_pure_svm():
    model = NestedCrossValidationModel(loader=BrainDataLoader(), estimator=SVMEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"model__C": np.linspace(0.01, 1, 5)})



def run_svm():
    model = NestedCrossValidationModel(loader=BrainDataLoader(), estimator=SVMWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 20), "model__C": np.linspace(0.01, 1, 5)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})

"""
Nested Grid Search Cross Validation on SVMWithPCAEstimator Complete on {'pca__n_components': range(4, 20), 'model__C': array([0.01  , 0.2575, 0.505 , 0.7525, 1.    ])}.

Weighted F1-score: 0.6708769270656748 (0.0452998459662532)
Balanced Accuracy: 0.5952424242424244 (0.03940540043069942)
Accuracy: 0.7284845817114725 (0.06491236887183817)
Sensitivity: 0.7190476190476189 (0.04037179191295001)
Specificity: 0.24848484848484845 (0.07459717348540104)
"""


def run_gpc():
    model = NestedCrossValidationModel(loader=BrainDataLoader(), estimator=GPCWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 20)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})


"""
Nested Grid Search Cross Validation on GPCWithPCAEstimator Complete on {'pca__n_components': range(4, 20)}.

Weighted F1-score: 0.6494607129762029 (0.0622867723028598)
Balanced Accuracy: 0.572469696969697 (0.06917767013938587)
Accuracy: 0.6886004056795131 (0.09823984489293379)
Sensitivity: 0.6906349206349207 (0.06638337390161679)
Specificity: 0.24393939393939396 (0.09418345616921246)
"""


def run_knn():
    model = NestedCrossValidationModel(loader=BrainDataLoader(), estimator=KNNWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 20), "model__n_neighbors": range(1, 16)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})

"""
Nested Grid Search Cross Validation on KNNWithPCAEstimator Complete on {'pca__n_components': range(4, 20), 'model__n_neighbors': range(1, 16)}.

Weighted F1-score: 0.6157401643215309 (0.0615185874000435)
Balanced Accuracy: 0.5812121212121213 (0.08571111815827179)
Accuracy: 0.6347322357724835 (0.076042131790083)
Sensitivity: 0.6122222222222222 (0.05796933186292484)
Specificity: 0.4924242424242424 (0.18875650305936287)
"""


def run_tree():
    model = NestedCrossValidationModel(loader=BrainDataLoader(), estimator=TreeWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 20)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})


"""
Nested Grid Search Cross Validation on TreeWithPCAEstimator Complete on {'pca__n_components': range(4, 20)}.

Weighted F1-score: 0.6337874154905297 (0.06542951660764956)
Balanced Accuracy: 0.5934696969696969 (0.05304057043461528)
Accuracy: 0.6518236423936516 (0.05151311826296007)
Sensitivity: 0.6342857142857142 (0.07598749416408838)
Specificity: 0.47727272727272724 (0.10475385465117336)
"""


def run_xgboost():
    model = NestedCrossValidationModel(loader=BrainDataLoader(clean_data=True), estimator=XGBoostEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({})

"""
Weighted F1-score: 0.6643127325779833 (0.03775917761435092)
Balanced Accuracy: 0.5983484848484849 (0.04324709048744225)
Accuracy: 0.6795164772080481 (0.050919285146733134)
Sensitivity: 0.6855555555555555 (0.0470237927801189)
Specificity: 0.353030303030303 (0.10449054333070705)
"""


def run_logistic():
    model = NestedCrossValidationModel(loader=BrainDataLoader(clean_data=True), estimator=LogisticEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({})

"""
Nested Grid Search Cross Validation on LogisticEstimator Complete on {}.

Weighted F1-score: 0.574870169700931 (0.08605705633827809)
Balanced Accuracy: 0.5124090909090908 (0.09059705792915028)
Accuracy: 0.5805572352274828 (0.08051149209031667)
Sensitivity: 0.5946031746031746 (0.10121882578290334)
Specificity: 0.2818181818181818 (0.15617089365727424)
"""


if __name__ == '__main__':
    run_pure_svm()