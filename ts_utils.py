import numpy as np
import pandas as pd


# base_formula should be lambda
# add missing values possibility
def create_time_series(base_formula, length, step=.1, noise_mean=0, noise_variance=1):
    x = np.arange(length * step, step=step)
    noise = np.random.randn(length) * noise_variance + noise_mean
    y = base_formula(x) + noise
    return pd.Series(y)


# noise must be included in a formula
def create_time_series_advanced(formula, length, step=.1):
    x = np.arange(length * step, step=step)
    return pd.Series(formula(x))


def df_sequential_split(df, train_ratio=.8):
    train_length = int(np.floor(len(df) * train_ratio))
    
    df_train = df.iloc[:train_length, :]
    df_test = df.iloc[train_length:, :]
    
    X_train = df_train.drop(df_train.columns[-1], axis=1)
    X_test = df_test.drop(df_test.columns[-1], axis=1)
    
    y_train = df_train.iloc[:, -1]
    y_test = df_test.iloc[:, -1]
    
    return X_train, X_test, y_train, y_test
