from DataAnalyzer.PerformanceAnalyzer import PerformanceAnalyzer
from DataLoader.CombinedDataLoader import CombinedDataLoader
from Estimator.GPCWithPCAEstimator import GPCWithPCAEstimator
from Estimator.KNNWithPCAEstimator import KNNWithPCAEstimator
from Estimator.TreeWithPCAEstimator import TreeWithPCAEstimator
from Model.NestedCrossValidationModel import NestedCrossValidationModel
from Estimator.MLPWithPCAEstimator import MLPWithPCAEstimator
from Estimator.SVMWithPCAEstimator import SVMWithPCAEstimator
import numpy as np


def run_mlp():
    model = NestedCrossValidationModel(loader=CombinedDataLoader(), estimator=MLPWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    # model.process({"pca__n_components": range(4, 16), "model__alpha": np.linspace(0.01, 1, 5)})
    model.process({"pca__n_components": range(4, 36), "model__alpha": [0.01, 0.25, 0.5, 1.0]})


"""
Nested Grid Search Cross Validation on MLPWithPCAEstimator Complete on {'pca__n_components': range(4, 36), 'model__alpha': [0.01, 0.25, 0.5, 1.0]}.

Weighted F1-score: 0.6982429785378386 (0.09692128126751108)
Balanced Accuracy: 0.6518326118326117 (0.09572520278013064)
Accuracy: 0.7265141310302601 (0.07596323090746226)
Sensitivity: 0.7161290322580645 (0.09655890030384366)
Specificity: 0.49111111111111116 (0.2395881239456632)
"""


def run_svm():
    model = NestedCrossValidationModel(loader=CombinedDataLoader(), estimator=SVMWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 36), "model__C": np.linspace(0.01, 1, 5)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})

"""
Nested Grid Search Cross Validation on SVMWithPCAEstimator Complete on {'pca__n_components': range(4, 36), 'model__C': array([0.01  , 0.2575, 0.505 , 0.7525, 1.    ])}.

Weighted F1-score: 0.7321875026019073 (0.05117517656913588)
Balanced Accuracy: 0.6662049062049061 (0.03557671991488441)
Accuracy: 0.7619495647721456 (0.058936936136218955)
Sensitivity: 0.7483870967741935 (0.06255070783763003)
Specificity: 0.4644444444444444 (0.1430531283504985)
Extra Parameters: {'params': {'model__C': 1.0, 'pca__n_components': 30}, 'score': 0.6460084033613445}
Extra Parameters: {'params': {'model__C': 1.0, 'pca__n_components': 18}, 'score': 0.6436974789915967}
Extra Parameters: {'params': {'model__C': 1.0, 'pca__n_components': 6}, 'score': 0.6027544351073764}
Extra Parameters: {'params': {'model__C': 1.0, 'pca__n_components': 16}, 'score': 0.6556722689075629}
Extra Parameters: {'params': {'model__C': 0.7525, 'pca__n_components': 4}, 'score': 0.7114729225023342}
"""


def run_gpc():
    model = NestedCrossValidationModel(loader=CombinedDataLoader(), estimator=GPCWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 36)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})


"""
Nested Grid Search Cross Validation on GPCWithPCAEstimator Complete on {'pca__n_components': range(4, 36)}.

Weighted F1-score: 0.6656511999369141 (0.007459275027861918)
Balanced Accuracy: 0.6311399711399711 (0.05234437960148931)
Accuracy: 0.6990638348457423 (0.04444519313142175)
Sensitivity: 0.6709677419354838 (0.024139725075960922)
Specificity: 0.531111111111111 (0.23521988662043095)
Extra Parameters: {'params': {'pca__n_components': 5}, 'score': 0.6643907563025209}
Extra Parameters: {'params': {'pca__n_components': 23}, 'score': 0.671218487394958}
Extra Parameters: {'params': {'pca__n_components': 21}, 'score': 0.6249183006535948}
Extra Parameters: {'params': {'pca__n_components': 16}, 'score': 0.6874299719887954}
Extra Parameters: {'params': {'pca__n_components': 15}, 'score': 0.6998482726423904}
"""


def run_knn():
    model = NestedCrossValidationModel(loader=CombinedDataLoader(), estimator=KNNWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 36), "model__n_neighbors": range(1, 16)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})

"""
Weighted F1-score: 0.6961976314260296 (0.11091920260765174)
Balanced Accuracy: 0.6843434343434343 (0.08947133764805083)
Accuracy: 0.7366048022743231 (0.0782293439895533)
Sensitivity: 0.6903225806451612 (0.11648690377592123)
Specificity: 0.6777777777777778 (0.10517475169954874)
Extra Parameters: {'params': {'model__n_neighbors': 6, 'pca__n_components': 30}, 'score': 0.7291316526610643}
Extra Parameters: {'params': {'model__n_neighbors': 6, 'pca__n_components': 15}, 'score': 0.7272642390289448}
Extra Parameters: {'params': {'model__n_neighbors': 4, 'pca__n_components': 6}, 'score': 0.7026610644257704}
Extra Parameters: {'params': {'model__n_neighbors': 6, 'pca__n_components': 21}, 'score': 0.7347572362278244}
Extra Parameters: {'params': {'model__n_neighbors': 6, 'pca__n_components': 11}, 'score': 0.7640172735760971}
"""


def run_tree():
    model = NestedCrossValidationModel(loader=CombinedDataLoader(), estimator=TreeWithPCAEstimator(),
                                       analyzer=PerformanceAnalyzer())
    model.process({"pca__n_components": range(4, 36)})
    # model.process({"pca__n_components": [11, 12], "model__alpha": [0.25, 0.5]})


"""
Nested Grid Search Cross Validation on TreeWithPCAEstimator Complete on {'pca__n_components': range(4, 36)}.

Weighted F1-score: 0.7020101820892023 (0.06757839977019416)
Balanced Accuracy: 0.6271139971139971 (0.05637912821325474)
Accuracy: 0.7099613928225429 (0.07736196515736556)
Sensitivity: 0.7161290322580645 (0.07468281872767886)
Specificity: 0.40444444444444444 (0.07522804835471447)
Extra Parameters: {'params': {'pca__n_components': 23}, 'score': 0.6892857142857143}
Extra Parameters: {'params': {'pca__n_components': 17}, 'score': 0.6718720821661999}
Extra Parameters: {'params': {'pca__n_components': 6}, 'score': 0.6053688141923436}
Extra Parameters: {'params': {'pca__n_components': 13}, 'score': 0.6786181139122315}
Extra Parameters: {'params': {'pca__n_components': 24}, 'score': 0.7602240896358544}
"""

if __name__ == '__main__':
    run_tree()