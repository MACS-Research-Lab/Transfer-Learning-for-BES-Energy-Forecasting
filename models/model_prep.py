import numpy as np
import pandas as pd


def df_to_3d(lstm_datadf, num_columns, step_back):
    vec = np.empty((lstm_datadf.shape[0], step_back, num_columns))
    for sample_index, (dfindex, sample) in enumerate(lstm_datadf.iterrows()):
        # Access row values using column names
        for tsindex, timestep in enumerate(range(1, step_back + 1)):
            if timestep == 0:
                continue
            else:
                suffix = f"(t-{timestep})"
            cur_ts_values = [
                value for (col_name, value) in sample.items() if suffix in col_name
            ]
            vec[sample_index][tsindex] = cur_ts_values
            # print(f"{suffix}: {vec[sample_index][tsindex]} which is supposed to be {cur_ts_values}")
    return vec


def create_timesteps(df, target_names, n_in=1, n_out=1):
    """
    Frame a time series as a supervised learning dataset.
    Arguments:
    data: Sequence of observations as a list or NumPy array.
    n_in: Number of lag observations as input (X).
    n_out: Number of observations as output (y).
    dropnan: Boolean whether or not to drop rows with NaN values.
    target_names: List of target variables to be predicted
    Returns:
    Pandas DataFrame of series framed for supervised learning.
    """
    n_vars = df.shape[1]
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f"{df.columns[j]}(t-{i})") for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df[target_names].shift(-i))
        if i == 0:
            names += [
                (f"{df.columns[j]}(t)")
                for j in range(n_vars)
                if df.columns[j] in target_names
            ]
        else:
            names += [
                (f"{df.columns[j]}(t+{i})")
                for j in range(n_vars)
                if df.columns[j] in target_names
            ]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # set rows with NaN values to zero
    for col in agg.columns:
        if pd.api.types.is_numeric_dtype(agg[col]):
            agg[col].fillna(0, inplace=True)
        else:
            agg[col].fillna(False, inplace=True)

    return agg
