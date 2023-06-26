# Load Packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px

import math
from datetime import datetime
import calendar
import pytz

import warnings

warnings.filterwarnings("ignore")

from typing import List

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import keras

import model_prep

datapath = "../data"

step_back = 6  # window size = 6*5 = 30 mins
season_map = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
    "winter": [12, 1, 2],
}
year = 2022


def create_model(
    building_name: str,
    tower_number: int,
    season: str,
    features: List[str],
    target: str,
    plot_history: bool = False,
    train_percentage: float = 0.75,
    use_delta: bool = True,
):
    """
    1. Convert data into a model-compatible shape
    """

    lstm_df, scaler, first_temp = create_preprocessed_lstm_df(
        building_name=building_name,
        tower_number=tower_number,
        features=features,
        target=target,
        season=season,
        use_delta=use_delta,
    )

    """
    2. Split data into training and testing sets
    """

    X = lstm_df.drop(f"{target}(t)", axis=1)  # drop target column
    y = lstm_df[f"{target}(t)"]  # only have target column

    # split into input and outputs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_percentage), shuffle=False
    )

    """
    3. Get timestepped data as a 3D vector
    """
    vec_X_train = model_prep.df_to_3d(
        lstm_dtframe=X_train, num_columns=len(features) + 1, step_back=step_back
    )
    vec_X_test = model_prep.df_to_3d(
        lstm_dtframe=X_test, num_columns=len(features) + 1, step_back=step_back
    )

    vec_y_train = y_train.values
    vec_y_test = y_test.values

    # print(vec_X_train.shape, vec_X_test.shape, vec_y_train.shape, vec_y_test.shape)

    """
    4. Create and Train model
    """
    model = keras.models.Sequential()
    model.add(
        keras.layers.LSTM(50, input_shape=(vec_X_train.shape[1], vec_X_train.shape[2]))
    )
    model.add(keras.layers.Dense(1))
    model.compile(loss="mae", optimizer="adam")

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
        plt.plot(history.history["loss"], label="train")
        plt.legend()
        plt.show()

    # make predictions
    yhat = model.predict(vec_X_test)

    """
    5. Display results
    """
    results_df = pd.DataFrame(
        {
            "actual": vec_y_test.reshape((vec_y_test.shape[0])),
            "predicted": yhat.reshape((yhat.shape[0])),
        },
        index=y_test.index,
    )
    results_df = scaler.denormalize_results(results_df)
    if use_delta:
        results_df["actual"] = results_df["actual"] + first_temp
        results_df["predicted"] = results_df["predicted"] + first_temp

    # Create a new DataFrame with the desired 5-minute interval index
    new_index = pd.date_range(
        start=pytz.utc.localize(datetime(year, season_map[season][0], 1)),
        end=pytz.utc.localize(
            datetime(
                year,
                season_map[season][-1],
                calendar.monthrange(year, season_map[season][-1])[1],
            )
        ),
        freq="5min",
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


def intra_season_transfer(
    from_building_name: str,
    from_tower_number: int,
    to_building_name: str,
    to_tower_number: int,
    to_features: List[str],
    to_target: str,
    to_season: str,
    from_season: str = None,
    finetuning_percentage: float = 0,
    finetune_epochs: int = 10,
    finetune_plot_history: bool = False,
    display_results: bool = True,
    use_delta: bool = True,
):
    # fix inputs
    if from_season == None:
        from_season = to_season

    """
    1. Load data and do LSTM preprocessing
    """

    lstm_to_df, to_scaler, to_first_temp = create_preprocessed_lstm_df(
        building_name=to_building_name,
        tower_number=to_tower_number,
        features=to_features,
        target=to_target,
        season=to_season,
        use_delta=use_delta,
    )
    print(f"Tower {to_tower_number} first temp: {to_first_temp}")

    """
    2. Convert tower data into a model-compatible shape i.e. get timestepped data as a 3D vector
    """

    X = lstm_to_df.drop(f"{to_target}(t)", axis=1)  # drop target column
    y = lstm_to_df[f"{to_target}(t)"]  # only have target column

    # if no finetuning is required
    if finetuning_percentage == 0:
        # entire set is for testing
        X_test = X
        y_test = y

        # create 3d vector form of data
        vec_X_test = model_prep.df_to_3d(
            lstm_dtframe=X_test, num_columns=len(to_features) + 1, step_back=step_back
        )
        vec_y_test = y_test.values
        # print(vec_X_test.shape, vec_y_test.shape)

        # load model
        model = keras.models.load_model(
            f"../models_saved/{from_building_name.lower()}{from_tower_number}_{from_season}_lstm/"
        )

    # if finetuning is required
    else:
        # split train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=(1 - finetuning_percentage), shuffle=False
        )

        # create 3d vector form of data
        vec_X_train = model_prep.df_to_3d(
            lstm_dtframe=X_train, num_columns=len(to_features) + 1, step_back=step_back
        )
        vec_X_test = model_prep.df_to_3d(
            lstm_dtframe=X_test, num_columns=len(to_features) + 1, step_back=step_back
        )

        vec_y_train = y_train.values
        vec_y_test = y_test.values

        print(
            f"finetuning_percentage: {finetuning_percentage} vec_X_train.shape: {vec_X_train.shape}, vec_X_test.shape: {vec_X_test.shape}, vec_y_train.shape: {vec_y_train.shape}, vec_y_test.shape: {vec_y_test.shape}"
        )

        # load and finetune model
        base_model = keras.models.load_model(
            f"../models_saved/{from_building_name.lower()}{from_tower_number}_{from_season}_lstm/"
        )
        model = finetune(
            model=base_model,
            training_feature_vec=vec_X_train,
            training_target_vec=vec_y_train,
            epochs=finetune_epochs,
            plot_history=finetune_plot_history,
        )

    """
    3. Load model, finetune and predict
    """

    yhat = model.predict(vec_X_test)

    """
    4. Display results
    """

    # save results
    results_df = pd.DataFrame(
        {
            "actual": vec_y_test.reshape((vec_y_test.shape[0])),
            "predicted": yhat.reshape((yhat.shape[0])),
        },
        index=y_test.index,
    )
    print(results_df)
    results_df = to_scaler.denormalize_results(results_df)
    print(results_df)
    if use_delta:
        results_df["actual"] = results_df["actual"] + to_first_temp
        results_df["predicted"] = results_df["predicted"] + to_first_temp

    rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["predicted"]))

    # display results
    def display_transfer_results():
        # Create a new DataFrame with the desired 5-minute interval index, and merge the new DataFrame with the original DataFrame
        display_df = pd.DataFrame(
            index=pd.date_range(
                start=results_df.index.min(), end=results_df.index.max(), freq="5min"
            )
        ).merge(results_df, how="left", left_index=True, right_index=True)

        print("RMSE: %.3f" % rmse)

        fig = px.line(display_df, x=display_df.index, y=["actual", "predicted"])
        fig.update_layout(
            title=f"{from_building_name} Tower {from_tower_number} {from_season} model used on {to_building_name} Tower {to_tower_number} {to_season} ({finetuning_percentage*100}% fine-tuning) LSTM Model Results",
            xaxis_title="time",
            yaxis_title=to_target,
        )
        fig.show()

        if from_building_name == to_building_name:
            if from_season != to_season:
                fig.write_html(
                    f"../plots/intrabuilding_transfers/interseason/{from_building_name.lower()}{from_tower_number}{from_season}_to_{to_building_name.lower()}{to_tower_number}{to_season}_{finetuning_percentage}_lstm.html"
                )
            else:
                fig.write_html(
                    f"../plots/intrabuilding_transfers/{from_building_name.lower()}{from_tower_number}_to_{to_building_name.lower()}{to_tower_number}_{to_season}_{finetuning_percentage}_lstm.html"
                )
        else:
            if from_season != to_season:
                fig.write_html(
                    f"../plots/interbuilding_transfers/interseason/{from_building_name.lower()}{from_tower_number}{from_season}_to_{to_building_name.lower()}{to_tower_number}{to_season}_{finetuning_percentage}_lstm.html"
                )
            else:
                fig.write_html(
                    f"../plots/interbuilding_transfers/{from_building_name.lower()}{from_tower_number}_to_{to_building_name.lower()}{to_tower_number}_{finetuning_percentage}_{to_season}_lstm.html"
                )

    if display_results:
        display_transfer_results()

    return rmse


def create_preprocessed_lstm_df(
    building_name: str,
    tower_number: int,
    features: List[str],
    target: str,
    season: str,
    use_delta: bool = True,
):
    """
    1. Load data and do LSTM preprocessing
    """
    # load data
    dtframe = pd.read_csv(
        f"../data/{building_name.lower()}/{building_name.lower()}_tower_{tower_number}_preprocessed.csv",
        index_col="time",
    )
    dtframe.index = pd.to_datetime(dtframe.index)

    # only take data for one season
    dtframe = model_prep.choose_season(
        dtframe,
        season=season,
        season_col_name=f"{building_name}_Tower_{tower_number} season",
    )

    # save a boolean series that specifies whether the cooling tower is on
    on_condition = dtframe[f"{building_name}_Tower_{tower_number} fanStatus"]

    # select features and targets and create final dataframe that includes only relevant features and targets
    dtframe = dtframe[features].join(dtframe[target], on=dtframe.index)

    # if difference from first temperature should be used as for predictions then return the first temperature
    first_temp = dtframe.iloc[0, dtframe.columns.get_loc(target)]
    if use_delta:
        dtframe[target] = dtframe[target] - first_temp

    # normalize data
    scaler = model_prep.NormalizationHandler()
    dtframe = scaler.normalize(dtframe=dtframe, target_col=target)

    # prepare dataframe for lstm by adding timesteps
    lstm_dtframe = model_prep.create_timesteps(
        dtframe, n_in=step_back, n_out=1, target_name=target
    )

    # remove cases where spring data would leak into summer data (i.e. intial timesteps)
    lstm_dtframe = model_prep.remove_irrelevant_data(
        lstm_dtframe, on_condition, step_back
    )
    return lstm_dtframe, scaler, first_temp


def finetune(
    model: keras.engine.sequential.Sequential,
    training_feature_vec: np.ndarray,
    training_target_vec: np.ndarray,
    epochs: int = 10,
    plot_history: bool = False,
):
    model.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
        loss="mae",
        metrics=[keras.metrics.BinaryAccuracy()],
    )

    history = model.fit(
        training_feature_vec,
        training_target_vec,
        epochs=epochs,
        verbose=0,
        shuffle=False,
    )

    # plot history
    if plot_history:
        plt.plot(history.history["loss"], label="train")
        plt.legend()
        plt.show()

    return model
