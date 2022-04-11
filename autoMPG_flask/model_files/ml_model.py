
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

### functions

def preprocess_origin_col(df):
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA",3: "Germany"})
    return df


class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def __init__(self, acceleration_on_horsepower=True):
        self.acceleration_on_horsepower = acceleration_on_horsepower

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        acceleration_ind = 4
        horsepower_ind = 2
        cylinders_ind = 0
        acceleration_on_cylinders = X[:, acceleration_ind] / X[:, cylinders_ind]
        if self.acceleration_on_horsepower:
            acceleration_on_horsepower = X[:, acceleration_ind] / X[:, horsepower_ind]
            # np.c_ method concatenate the arrays
            return np.c_[X, acceleration_on_horsepower, acceleration_on_cylinders]

        return np.c_[X, acceleration_on_cylinders]


def num_pipeline_transformer(data):
    """
    Function to process numerical transformations

    Argument:
        data: original dataframe
    Returns:
        num_attrs: numerical dataframe
        num_pipleline: numerical pipeline object
    """
    numerics = ['float64', 'int64']

    num_attrs = data.select_dtypes(include=numerics)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler())
    ])
    return num_attrs, num_pipeline


def pipeline_transformer(data):
    """
    Complete transformation pipeline for both
    numerical and categorical data.

    Argument:
        data: original dataframe.
    Returns:
        prepared_data: transformed data, ready to use.
    """
    cat_attrs = ["Origin"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
    ])
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data

def predict_mpg(config, model):
    """
    Takes an instance process the data, applies the desired model and returns the prediction.
    :param config: input instances from a prediction is going to be made.
    :param model: Model that will be applied.
    :return: prediction.
    """
    if type(config) == dict:
        df = pd.DataFrame(config)
    else:
        df = config

    preproc_df = preprocess_origin_col(df)
    prepared_df = pipeline_transformer(preproc_df)
    y_pred = model.predict(prepared_df)
    return y_pred
