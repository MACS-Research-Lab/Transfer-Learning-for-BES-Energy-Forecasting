# Load Packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px

import warnings

warnings.filterwarnings("ignore")

from typing import List

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM

import model_prep

datapath = "../data"

step_back = 6  # window size = 6*5 = 30 mins


def create_model(
    building_name: str,
    tower_number: int,
    season: str,
    features: List[str],
    target: str,
    plot_history: bool = False,
):
    """
    1. Convert data into a model-compatible shape
    """

    # load data
    print("\nLoading data...")
    df = pd.read_csv(
        f"../data/{building_name.lower()}/{building_name.lower()}_tower_{tower_number}_preprocessed.csv",
        index_col="time",
    )
    df.index = pd.to_datetime(df.index)

    # only take data for one season
    print("\nSelecting season...")
    df = model_prep.choose_season(
        df,
        season=season,
        season_col_name=f"{building_name}_Tower_{tower_number} season",
    )

    # save a boolean series that specifies whether the cooling tower is on
    on_condition = df[f"{building_name}_Tower_{tower_number} fanStatus"]

    # select features and target and create final dataframe that includes only relevant features and target
    print("\nSelecting target and features...")
    df = df[features].join(df[target], on=df.index)

    # normalize data
    print("\nNormalizing...")
    scaler = model_prep.NormalizationHandler()
    df = scaler.normalize(dtframe=df, target_col=target)

    # prepare dataframe for lstm by adding timesteps
    print("\nCreating timesteps...")
    lstm_df = model_prep.create_timesteps(
        df, n_in=step_back, n_out=1, target_name=target
    )

    # remove cases where spring data would leak into summer data (i.e. intial timesteps)
    print("\nRemoving irrelevant data...")
    lstm_df = model_prep.remove_irrelevant_data(lstm_df, on_condition, step_back)

    """
    2. Split data into training and testing sets
    """
    print("\nSplitting training and testing sets...")
    tss = TimeSeriesSplit(n_splits=3)

    X = lstm_df.drop(f"{target}(t)", axis=1)  # drop target column
    y = lstm_df[f"{target}(t)"]  # only have target column

    for train_index, test_index in tss.split(X):  # split into training and testing
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    """
    3. Get timestepped data as a 3D vector
    """
    print("\nCreating 3D vector...")
    vec_X_train = model_prep.df_to_3d(
        lstm_dtframe=X_train, num_columns=len(features) + 1, step_back=step_back
    )
    vec_X_test = model_prep.df_to_3d(
        lstm_dtframe=X_test, num_columns=len(features) + 1, step_back=step_back
    )

    vec_y_train = y_train.values
    vec_y_test = y_test.values

    print(vec_X_train.shape, vec_X_test.shape, vec_y_train.shape, vec_y_test.shape)

    """
    4. Create and Train model
    """
    print("\nCreating model...")
    model = Sequential()
    model.add(LSTM(50, input_shape=(vec_X_train.shape[1], vec_X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss="mae", optimizer="adam")

    print("\nTraining model...")
    history = model.fit(
        vec_X_train,
        vec_y_train,
        epochs=50,
        batch_size=72,
        validation_data=(vec_X_test, vec_y_test),
        verbose=0,
        shuffle=False,
    )

    # plot history
    if plot_history:
        print("\nPlotting history...")
        plt.plot(history.history["loss"], label="train")
        plt.legend()
        plt.show()

    print("\nMaking predictions...")
    yhat = model.predict(vec_X_test)

    """
    5. Display results
    """
    print("\nResults:...")
    results_df = pd.DataFrame(
        {
            "actual": vec_y_test.reshape((vec_y_test.shape[0])),
            "predicted": yhat.reshape((yhat.shape[0])),
        },
        index=y_test.index,
    )
    results_df = scaler.denormalize_results(results_df)

    # Create a new DataFrame with the desired 5-minute interval index
    new_index = pd.date_range(
        start=results_df.index.min(), end=results_df.index.max(), freq="5min"
    )
    display_df = pd.DataFrame(index=new_index)
    # Merge the new DataFrame with the original DataFrame
    display_df = display_df.merge(
        results_df, how="left", left_index=True, right_index=True
    )

    mabs_error = mean_absolute_error(results_df["actual"], results_df["predicted"])
    rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["predicted"]))
    print("Mean Absolute Error: %.3f" % mabs_error)
    print("RMSE: %.3f" % rmse)

    fig = px.line(display_df, x=display_df.index, y=["actual", "predicted"])
    fig.update_layout(
        title=f"{building_name} Tower {tower_number} LSTM Model Results",
        xaxis_title="time",
        yaxis_title=target,
    )
    fig.show()

    fig.write_html(
        f"../plots/prepared_models/{building_name.lower()}{tower_number}_{season}_lstm.html"
    )
    model.summary()
    model.save(f"../models_saved/{building_name.lower()}{tower_number}_{season}_lstm/")


def intra_building_transfer(
    building_name: str,
    from_csv_path: str,
    tower_number: int,
    season: str,
    features: List[str],
    target: str,
    retraining_percentage: float = 0,
):
    # load data
    df = pd.read_csv(from_csv_path, index_col="time")
    df.index = pd.to_datetime(df.index)

    # only take data for one season
    df = model_prep.choose_season(
        df,
        season=season,
        season_col_name=f"{building_name}_Tower_{tower_number} season",
    )

    # save a boolean series that specifies whether the cooling tower is on
    on_condition = df[f"{building_name}_Tower_{tower_number} fanStatus"]

    # select features and targets and create final dataframe that includes only relevant features and targets
    df = df[features].join(df[target], on=df.index)

    # normalize data
    scaler = model_prep.NormalizationHandler()
    df = scaler.normalize(dtframe=df, target_col=target)

    # prepare dataframe for lstm by adding timesteps
    lstm_df = model_prep.create_timesteps(
        df, n_in=step_back, n_out=1, target_name=target
    )

    # remove cases where spring data would leak into summer data (i.e. intial timesteps)
    lstm_df = model_prep.remove_irrelevant_data(lstm_df, on_condition, step_back)

    """
    2. Convert tower data into a model-compatible shape i.e. get timestepped data as a 3D vector
    """

    tss = TimeSeriesSplit(n_splits=3)
    X = lstm_df.drop(f"{target}(t)", axis=1)  # drop target column
    y = lstm_df[f"{target}(t)"]  # only have target column

    vec_X_test = model_prep.df_to_3d(
        lstm_dtframe=X, num_columns=len(features) + 1, step_back=step_back
    )
    vec_y_test = y.values
    print(vec_X_test.shape, vec_y_test.shape)

    """
    3. Load model and predict
    """
    # load model of the other tower
    model = load_model(
        f"../models_saved/{building_name.lower()}{3-tower_number}_{season}_lstm/"
    )
    yhat = model.predict(vec_X_test)

    # display results
    results_df = pd.DataFrame(
        {
            "actual": vec_y_test.reshape((vec_y_test.shape[0])),
            "predicted": yhat.reshape((yhat.shape[0])),
        },
        index=y.index,
    )
    results_df = scaler.denormalize_results(results_df)

    # Create a new DataFrame with the desired 5-minute interval index
    new_index = pd.date_range(
        start=results_df.index.min(), end=results_df.index.max(), freq="5min"
    )
    display_df = pd.DataFrame(index=new_index)
    # Merge the new DataFrame with the original DataFrame
    display_df = display_df.merge(
        results_df, how="left", left_index=True, right_index=True
    )

    mabs_error = mean_absolute_error(results_df["actual"], results_df["predicted"])
    rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["predicted"]))
    print("Mean Absolute Error: %.3f" % mabs_error)
    print("RMSE: %.3f" % rmse)

    fig = px.line(display_df, x=display_df.index, y=["actual", "predicted"])
    fig.update_layout(
        title=f"{building_name} Tower {3-tower_number} model used on {building_name} Tower {tower_number} ({season}) (zero retraining) LSTM Model Results",
        xaxis_title="time",
        yaxis_title=target,
    )
    fig.show()

    fig.write_html(
        f"../plots/intrabuilding_transfers/{building_name.lower()}{3-tower_number}_to_{building_name.lower()}{tower_number}_{season}_lstm.html"
    )


def inter_building_transfer(
    from_building_name: str,
    from_tower_number: int,
    to_building_name: str,
    to_tower_number: int,
    to_features: List[str],
    to_target: str,
    season: str,
    retraining_percentage: float = 0,
):
    # load data
    to_df = pd.read_csv(
        f"../data/{to_building_name.lower()}/{to_building_name.lower()}_tower_{to_tower_number}_preprocessed.csv",
        index_col="time",
    )
    to_df.index = pd.to_datetime(to_df.index)

    # only take data for one season
    to_df = model_prep.choose_season(
        to_df,
        season=season,
        season_col_name=f"{to_building_name}_Tower_{to_tower_number} season",
    )

    # save a boolean series that specifies whether the cooling tower is on
    on_condition = to_df[f"{to_building_name}_Tower_{to_tower_number} fanStatus"]

    # select features and targets and create final dataframe that includes only relevant features and targets
    to_df = to_df[to_features].join(to_df[to_target], on=to_df.index)

    # normalize data
    scaler = model_prep.NormalizationHandler()
    to_df = scaler.normalize(dtframe=to_df, target_col=to_target)

    # prepare dataframe for lstm by adding timesteps
    lstm_to_df = model_prep.create_timesteps(
        to_df, n_in=step_back, n_out=1, target_name=to_target
    )

    # remove cases where spring data would leak into summer data (i.e. intial timesteps)
    lstm_to_df = model_prep.remove_irrelevant_data(lstm_to_df, on_condition, step_back)

    """
    2. Convert tower data into a model-compatible shape i.e. get timestepped data as a 3D vector
    """

    tss = TimeSeriesSplit(n_splits=3)
    X = lstm_to_df.drop(f"{to_target}(t)", axis=1)  # drop target column
    y = lstm_to_df[f"{to_target}(t)"]  # only have target column

    vec_X_test = model_prep.df_to_3d(
        lstm_dtframe=X, num_columns=len(to_features) + 1, step_back=step_back
    )
    vec_y_test = y.values
    print(vec_X_test.shape, vec_y_test.shape)

    """
    3. Load model and predict
    """
    # load model of the other tower
    model = load_model(
        f"../models_saved/{from_building_name.lower()}{from_tower_number}_{season}_lstm/"
    )
    yhat = model.predict(vec_X_test)

    # display results
    results_df = pd.DataFrame(
        {
            "actual": vec_y_test.reshape((vec_y_test.shape[0])),
            "predicted": yhat.reshape((yhat.shape[0])),
        },
        index=y.index,
    )
    results_df = scaler.denormalize_results(results_df)

    # Create a new DataFrame with the desired 5-minute interval index
    new_index = pd.date_range(
        start=results_df.index.min(), end=results_df.index.max(), freq="5min"
    )
    display_df = pd.DataFrame(index=new_index)
    # Merge the new DataFrame with the original DataFrame
    display_df = display_df.merge(
        results_df, how="left", left_index=True, right_index=True
    )

    mabs_error = mean_absolute_error(results_df["actual"], results_df["predicted"])
    rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["predicted"]))
    print("Mean Absolute Error: %.3f" % mabs_error)
    print("RMSE: %.3f" % rmse)

    fig = px.line(display_df, x=display_df.index, y=["actual", "predicted"])
    fig.update_layout(
        title=f"{from_building_name} Tower {from_tower_number} model used on {to_building_name} Tower {to_tower_number} ({season}) (zero retraining) LSTM Model Results",
        xaxis_title="time",
        yaxis_title=to_target,
    )
    fig.show()

    fig.write_html(
        f"../plots/interbuilding_transfers/{from_building_name.lower()}{from_tower_number}_to_{to_building_name.lower()}{to_tower_number}_{season}_lstm.html"
    )
