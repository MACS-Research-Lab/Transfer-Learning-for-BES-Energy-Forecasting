# Load Packages
import pandas as pd
import numpy as np
import math
from matplotlib import pyplot as plt
import plotly.express as px

import warnings

warnings.filterwarnings("ignore")

from typing import List

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

import keras

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
    train_percentage: float = 0.75,
):
    """
    1. Convert data into a model-compatible shape
    """

    lstm_df, scaler = create_preprocessed_lstm_df(
        building_name=building_name,
        tower_number=tower_number,
        features=features,
        target=target,
        season=season,
    )

    """
    2. Split data into training and testing sets
    """

    X = lstm_df.drop(f"{target}(t)", axis=1)  # drop target column
    y = lstm_df[f"{target}(t)"]  # only have target column

    train_split = math.ceil(train_percentage * len(X))

    # split into input and outputs
    X_train = X.iloc[:train_split, :]
    X_test = X.iloc[train_split:, :]
    y_train = y.iloc[:train_split]
    y_test = y.iloc[train_split:]

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


def intra_season_transfer(
    from_building_name: str,
    from_tower_number: int,
    to_building_name: str,
    to_tower_number: int,
    to_features: List[str],
    to_target: str,
    season: str,
    finetuning_percentage: float = 0,
    finetune_epochs: int = 10,
    finetune_plot_history: bool = False,
    displayResults: bool = True,
):
    """
    1. Load data and do LSTM preprocessing
    """

    lstm_to_df, scaler = create_preprocessed_lstm_df(
        building_name=to_building_name,
        tower_number=to_tower_number,
        features=to_features,
        target=to_target,
        season=season,
    )

    """
    2. Convert tower data into a model-compatible shape i.e. get timestepped data as a 3D vector
    """

    X = lstm_to_df.drop(f"{to_target}(t)", axis=1)  # drop target column
    y = lstm_to_df[f"{to_target}(t)"]  # only have target column

    train_split = math.ceil(finetuning_percentage * len(X))
    X_train = X.iloc[:train_split, :]
    X_test = X.iloc[train_split:, :]
    y_train = y.iloc[:train_split]
    y_test = y.iloc[train_split:]

    # if no finetuning is required
    if finetuning_percentage == 0:
        # create 3d vector form of data
        vec_X_test = model_prep.df_to_3d(
            lstm_dtframe=X_test, num_columns=len(to_features) + 1, step_back=step_back
        )
        vec_y_test = y_test.values
        # print(vec_X_test.shape, vec_y_test.shape)

        # load model
        model = keras.models.load_model(
            f"../models_saved/{from_building_name.lower()}{from_tower_number}_{season}_lstm/"
        )

    # if finetuning is required
    else:
        # create 3d vector form of data
        vec_X_train = model_prep.df_to_3d(
            lstm_dtframe=X_train, num_columns=len(to_features) + 1, step_back=step_back
        )
        vec_X_test = model_prep.df_to_3d(
            lstm_dtframe=X_test, num_columns=len(to_features) + 1, step_back=step_back
        )

        vec_y_train = y_train.values
        vec_y_test = y_test.values

        # print(vec_X_train.shape, vec_X_test.shape, vec_y_train.shape, vec_y_test.shape)

        # load and finetune model
        model = keras.models.load_model(
            f"../models_saved/{from_building_name.lower()}{from_tower_number}_{season}_lstm/"
        )
        model = finetune(
            model=model,
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
    results_df = scaler.denormalize_results(results_df)

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
            title=f"{from_building_name} Tower {from_tower_number} model used on {to_building_name} Tower {to_tower_number} ({season}) ({finetuning_percentage*100}% fine-tuning) LSTM Model Results",
            xaxis_title="time",
            yaxis_title=to_target,
        )
        fig.show()

        if from_building_name == to_building_name:
            fig.write_html(
                f"../plots/intrabuilding_transfers/{from_building_name.lower()}{from_tower_number}_to_{to_building_name.lower()}{to_tower_number}_{season}_{finetuning_percentage}_lstm.html"
            )
        else:
            fig.write_html(
                f"../plots/interbuilding_transfers/{from_building_name.lower()}{from_tower_number}_to_{to_building_name.lower()}{to_tower_number}_{finetuning_percentage}_{season}_lstm.html"
            )

    if displayResults:
        display_transfer_results()

    return rmse


def create_preprocessed_lstm_df(
    building_name: str, tower_number: int, features: List[str], target: str, season: str
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
    return lstm_dtframe, scaler


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
