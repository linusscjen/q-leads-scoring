import numpy as np
import pandas as pd


def impute_cols(x_train: pd.DataFrame, x_test: pd.DataFrame, cols: list, imputation: str):
    """Input missing values for specified columns of both dataframes with a specific value

    Args:
        x_train (pd.DataFrame): Training dataset
        x_test (pd.DataFrame): Test dataset
        cols (list): list of columns that are imputed with a specific value
        imputation (str): Methodology type
    """
    train = x_train.copy()
    test = x_test.copy()
    
    train[cols].fillna(imputation, inplace=True)
    test[cols].fillna(imputation, inplace=True)
    
    return train, test
