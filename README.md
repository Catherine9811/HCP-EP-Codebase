# HCP Analysis

## How to run

Use python scripts start with `RunXXX.py` to run models in this repo.

Use unit test scripts start with `TestXXX.py` to run test on different components.

## Framework design

`DataLoader`: Contains data pre-processing and post-processing scripts for different datasets.

`Estimator`: Contains manually wrapped scikit-learn style base estimator.

`DataAnalyzer`: Perform data analysis on the given dataset.

`Model`: Contains different evaluation presets given the dataset and estimator.

`Utility`: Not used yet.

`Data`: Raw data file storage path.