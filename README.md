# ROCKET. Time Series Prediction and Random Convolutions.

## Purpose of the study:
The main purpose of the study is to present a comparison
of different models for time series forecasting/regression.
Namely, ROCKET model is compared against
several competitors like: Moving Average (MA), XGBoost,
LSTM. They are compared based on
computational complexity and prediction quality.

The work is done within the context of Skoltech "Models of Sequential Data" course in 2022.

## Datasets:
Airline data is used in all the experiments. Monthly totals of international airline passengers, 1949 to 1960.
Dimensionality: univariate Series length: 144 Frequency: Monthly Number of cases: 1
This data shows an increasing trend, non-constant (increasing) variance and periodic, seasonal patterns.
(see https://www.sktime.org/en/stable/api_reference/auto_generated/sktime.datasets.load_airline.html)

## Environment:
Use `conda` package manager to install required python packages. Run the following command (while in the root of the repository):
```
conda env create -f environment.yml
```
This will create new environment named `msd_6_term` with all required packages already installed. You can install additional packages by running:
```
conda install <package name>
```

## Reproduction of the experimental results:
```
Run experiment.ipynb 
```
In any environment that can process notebooks
