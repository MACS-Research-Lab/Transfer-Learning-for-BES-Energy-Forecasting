from typing import List
import numpy as np
import pandas as pd
from datetime import timedelta
import json

rootpath = ".."


def choose_season(
    dtframe: pd.DataFrame,
    season: str,
    season_col_name: str = "Season",
    verbose: bool = False,
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
    dtframe: pd.DataFrame,
    on_condition: pd.Series,
    step_back: int,
    verbose: bool = False,
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
    return vec


def create_timesteps(df, target_name, n_in=1, n_out=1):
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
        cols.append(df[target_name].shift(-i))
        if i == 0:
            names += [
                (f"{df.columns[j]}(t)")
                for j in range(n_vars)
                if df.columns[j] == target_name
            ]
        else:
            names += [
                (f"{df.columns[j]}(t+{i})")
                for j in range(n_vars)
                if df.columns[j] == target_name
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


def create_preprocessed_lstm_df(
    building_name: str,
    tower_number: int,
    features: List[str],
    target: str,
    season: str = None,
    use_delta: bool = False,
    step_back: int = 6,
):
    """
    1. Load data and do LSTM preprocessing
    """
    # load data
    dtframe = pd.read_csv(
        f"{rootpath}/data/{building_name.lower()}/{building_name.lower()}{tower_number}_preprocessed.csv",
        index_col="time",
    )
    dtframe.index = pd.to_datetime(dtframe.index)

    # only take data for one season
    if season:
        dtframe = choose_season(dtframe, season=season)
    else:
        season = "allyear"

    # save a boolean series that specifies whether the cooling tower is on
    on_condition = dtframe[target] > 0
    print(
        "number of times the hvac is on (energy consumption is > zero)",
        on_condition.value_counts(),
    )

    # select features and targets and create final dataframe that includes only relevant features and targets
    dtframe = dtframe[features].join(dtframe[target], on=dtframe.index)

    # prepare dataframe for lstm by adding timesteps
    lstm_dtframe = create_timesteps(
        dtframe, n_in=step_back, n_out=1, target_name=target
    )

    # remove cases where spring data would leak into summer data (i.e. intial timesteps)
    lstm_dtframe = remove_irrelevant_data(lstm_dtframe, on_condition, step_back)

    # if difference from first value should be used as for predictions then return the first value
    modified_target_name = f"{target}(t)"
    first_val = lstm_dtframe.iloc[0, lstm_dtframe.columns.get_loc(modified_target_name)]
    if use_delta:
        lstm_dtframe[modified_target_name] = (
            lstm_dtframe[modified_target_name] - first_val
        )

    return lstm_dtframe, first_val


def save_base_errors(
    model_type,
    building_name,
    tower_number,
    season,
    rmse,
    mae,
    mae_sd,
    training_time: None,
):
    # the base model result may be needed to be stored for multiple transfers
    if model_type == "LD":
        model_types = ["weight_initialization_LSTMDense"]
    elif model_type == "autoLSTM":
        model_types = ["weight_initialization_AutoLSTM"]
    elif model_type == "GRU":
        model_types = ["weight_initialization_GRU"]
    elif model_type == "autoGRU":
        model_types = ["weight_initialization_AutoGRU"]
    else:
        raise ValueError("Invalid model type")

    # load results file
    result_filename = f"{rootpath}/results/result_data/transfer_results.json"
    with open(result_filename, "r") as f:
        data = json.load(f)

    for mt in model_types:
        # if that building doesn't exist
        if f"{building_name}{tower_number}_{season}" not in data[mt]:
            data[mt][f"{building_name}{tower_number}_{season}"] = {"base": {}}
        elif "base" not in data[mt][f"{building_name}{tower_number}_{season}"]:
            data[mt][f"{building_name}{tower_number}_{season}"]["base"] = {}

        data[mt][f"{building_name}{tower_number}_{season}"]["base"]["rmse"] = rmse
        data[mt][f"{building_name}{tower_number}_{season}"]["base"]["mae"] = mae
        data[mt][f"{building_name}{tower_number}_{season}"]["base"]["mae_sd"] = mae_sd
        data[mt][f"{building_name}{tower_number}_{season}"]["base"][
            "training_time"
        ] = training_time

    # update results file
    with open(result_filename, "w") as f:
        json.dump(data, f, indent=4)
