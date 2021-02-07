import pandas as pd
import numpy as np


def df_for_sliding_window_with_nans(ts, width):
    columns = [str(x) for x in np.arange(width)]
    df_temp = pd.DataFrame([], columns=columns)
    df_temp['0'] = ts
    for col in range(1, width):
        for row in range(len(ts) - col):
            df_temp.iloc[row, col] = df_temp.iloc[row + 1, col - 1]

    return df_temp


def df_for_sliding_window(ts, width):
    df_with_nans = df_for_sliding_window_with_nans(ts, width)

    # must delete incomplete rows (width - 1)
    return df_with_nans.iloc[:len(ts) - width + 1, :]