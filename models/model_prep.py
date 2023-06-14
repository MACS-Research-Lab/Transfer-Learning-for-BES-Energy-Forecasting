import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler


class NormalizationHandler(object):
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.min_actual = None
        self.max_actual = None

    def normalize(self, dtframe: pd.DataFrame, target_col: str):
        self.min_actual = np.min(dtframe[target_col])
        self.max_actual = np.max(dtframe[target_col])
        print(f"Min = {self.min_actual}")
        print(f"Max = {self.max_actual}")

        scaled_df = pd.DataFrame(
            self.scaler.fit_transform(dtframe.values.astype("float32")),
            columns=dtframe.columns,
            index=dtframe.index,
        )
        return scaled_df

    def denormalize_results(self, results_dtframe: pd.DataFrame):
        for column in results_dtframe.columns:
            results_dtframe[column] = (
                results_dtframe[column] * (self.max_actual - self.min_actual)
                + self.min_actual
            )
        return results_dtframe


def choose_season(
    dtframe: pd.DataFrame, season: str, season_col_name: str, verbose: bool = True
):
    """
    Include only the data from a selected season.
    Arguments:
    dtframe (pandas.Dataframe): dataframe of ESB data with a season column containing values from ['summer', 'fall', 'winter', 'spring']
    season (str): Name of the season required.
    verbose (bool): Whether to print logs or not
    Returns:
    Pandas DataFrame containing only data for that season
    """
    if season not in ["summer", "fall", "winter", "spring"]:
        raise ValueError("Wrong season provided")
    dtframe = dtframe[dtframe[season_col_name] == season]
    if verbose:
        print(f"There are {dtframe.shape[0]} rows of data for the {season} season.")
    return dtframe


def remove_irrelevant_data(
    dtframe: pd.DataFrame, on_condition: pd.Series, step_back: int, verbose: bool = True
):
    """
    Remove cases where the cooling tower was off, or had timestep data from a time when it was off/considered a different season
    Arguments:
    dtframe (pandas.Dataframe): dataframe of ESB data that already has timestep columns
    on_condition (pandas.Series): boolean Series of True/False values for if the tower is on
    step_back (int): Window size for past data used in the LSTM.
    verbose (bool): Whether to print logs or not
    Returns:
    Pandas DataFrame ready for the LSTM.
    """
    initial_size = dtframe.shape[0]

    # remove cases where previous season's data could leak into current seasons' data (i.e. intial timesteps)
    dtframe = dtframe.iloc[step_back:]

    # remove cases in which tower was OFF, and cases where OFF data would be included in past timesteps of ON data
    to_drop = dtframe[~on_condition].index
    for multiplier in range(1, step_back + 1):
        to_drop = to_drop.union(
            dtframe[~on_condition].index + timedelta(minutes=5 * multiplier)
        )
    dtframe = dtframe.drop(dtframe.index.intersection(to_drop), axis=0)

    if verbose:
        print(
            f"Number of samples in summer data before removing off times: {initial_size}\n",
            f"Number of samples in summer data after removing off times: {dtframe.shape[0]}",
        )

    return dtframe


def df_to_3d(lstm_dtframe, num_columns, step_back):
    vec = np.empty((lstm_dtframe.shape[0], step_back, num_columns))
    for sample_index, (dfindex, sample) in enumerate(lstm_dtframe.iterrows()):
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
