# Load Packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px

import warnings

warnings.filterwarnings("ignore")

from typing import List

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

import keras
import time

rootpath = ".."
import model_prep

step_back = 6  # window size = 6*5 = 30 mins
season_map = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
    "winter": [12, 1, 2],
}


def create_base_eld_model(
    building_name: str,
    tower_number: int,
    features: List[str],
    target: str,
    season: str = None,
    train_percentage: float = 0.75,
    use_delta: bool = True,
    shuffle_seed: int = 42,
):
    """
    1. Convert data into a model-compatible shape
    """

    lstm_df, first_temp = model_prep.create_preprocessed_lstm_df(
        building_name=building_name,
        tower_number=tower_number,
        features=features,
        target=target,
        season=season,
        use_delta=use_delta,
    )
    if not season:
        season = "allyear"

    """
    2. Split data into training and testing sets
    """

    X = lstm_df.drop(f"{target}(t)", axis=1)  # drop target column
    y = lstm_df[f"{target}(t)"]  # only have target column

    # split into input and outputs
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(1 - train_percentage), shuffle=True, random_state=shuffle_seed
    )

    # scale feature data
    scaler = MinMaxScaler().fit(X_train)
    X_train[X_train.columns] = scaler.transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)

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

    """
    4. Create and Train model
    """
    # Define input shape for the autoencoder
    timesteps = step_back
    input_dim = len(features) + 1
    input_shape = (timesteps, input_dim)

    # selected hyperparameters
    dropout_rate = 0.0
    weight_constraint = keras.constraints.MaxNorm(7.0)
    # lstmcells = ?
    activation = "linear"
    neurons = 128
    optimizer = "Adagrad"

    # Define the autoencoder architecture using model.add()
    model = keras.models.Sequential()

    # Dense encoder layer
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    model.add(
        keras.layers.Dense(
            units=neurons,
            activation=activation,
            kernel_constraint=weight_constraint,
        )
    )
    model.add(keras.layers.Dropout(dropout_rate))  # Dropout after the encoder layer
    # LSTM bottleneck layer
    model.add(
        keras.layers.LSTM(
            units=32,
            activation=activation,
            recurrent_dropout=dropout_rate,
            kernel_constraint=weight_constraint,
        )
    )
    # Dense decoder layer
    model.add(
        keras.layers.Dense(
            units=neurons,
            activation=activation,
            kernel_constraint=weight_constraint,
        )
    )
    model.add(keras.layers.Dropout(dropout_rate))  # Dropout after the decoder layer
    # Output layer to predict the next timestep
    model.add(keras.layers.Dense(units=1))
    # Compile the autoencoder model
    model.compile(optimizer=optimizer, loss="mse")

    # Print the summary of the autoencoder model
    model.summary()

    start_time = time.time()
    history = model.fit(
        vec_X_train,
        vec_y_train,
        epochs=50,
        batch_size=72,
        validation_data=(vec_X_test, vec_y_test),
        verbose=0,
        shuffle=False,
    )
    end_time = time.time()
    training_time = end_time - start_time

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

    if use_delta:
        results_df["actual"] = results_df["actual"] + first_temp
        results_df["predicted"] = results_df["predicted"] + first_temp

    # SAVE ERROR AND DATA AVAILABILITY INFORMATION

    # calculate, display and save error results
    mae = mean_absolute_error(results_df["actual"], results_df["predicted"])
    rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["predicted"]))
    mae_sd = np.std(np.abs(results_df["actual"] - results_df["predicted"]))
    model_prep.save_base_errors(
        "autoLSTM",
        building_name,
        tower_number,
        season,
        rmse,
        mae,
        mae_sd,
        training_time,
    )

    # GENERATE PLOTS

    # Create a new DataFrame with the desired 5-minute interval index, and merge the new DataFrame with the original DataFrame
    display_df = pd.DataFrame(
        index=pd.date_range(
            start=results_df.index.min(), end=results_df.index.max(), freq="5min"
        )
    ).merge(results_df, how="left", left_index=True, right_index=True)

    # display trend of temperature predictions (actual vs predicted)
    fig = px.line(display_df, x=display_df.index, y=["actual", "predicted"])
    fig.update_layout(
        title=f"{building_name} Tower {tower_number} LSTM Autoencoder Model Results",
        xaxis_title="time",
        yaxis_title=target,
    )
    fig.show()
    fig.write_html(
        f"{rootpath}/results/plots/prepared_models/{building_name.lower()}_{season}_eld.html"
    )

    # save the model
    print(model.summary())
    model.save(
        f"{rootpath}/results/models_saved/base_models/{building_name.lower()}{tower_number}_{season}_eld/"
    )
