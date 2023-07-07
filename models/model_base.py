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

import model_prep

rootpath = ".."

step_back = 6  # window size = 6*5 = 30 mins
season_map = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
    "winter": [12, 1, 2],
}


def create_base_model(
    building_name: str,
    tower_number: int,
    features: List[str],
    target: str,
    season: str = None,
    plot_history: bool = False,
    train_percentage: float = 0.75,
    use_delta: bool = True,
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
        X, y, test_size=(1 - train_percentage), shuffle=False
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

    # print(vec_X_train.shape, vec_X_test.shape, vec_y_train.shape, vec_y_test.shape)

    """
    4. Create and Train model
    """
    model = keras.models.Sequential()
    model.add(
        keras.layers.LSTM(
            50,
            input_shape=(vec_X_train.shape[1], vec_X_train.shape[2]),
            bias_regularizer=keras.regularizers.L1(0.01),
        )
    )
    model.add(keras.layers.Dense(1, kernel_regularizer=keras.regularizers.L1(0.01)))
    model.compile(loss="mse", optimizer="adam")

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

    if use_delta:
        results_df["actual"] = results_df["actual"] + first_temp
        results_df["predicted"] = results_df["predicted"] + first_temp

    # Create a new DataFrame with the desired 5-minute interval index
    # Create a new DataFrame with the desired 5-minute interval index, and merge the new DataFrame with the original DataFrame
    display_df = pd.DataFrame(
        index=pd.date_range(
            start=results_df.index.min(), end=results_df.index.max(), freq="5min"
        )
    ).merge(results_df, how="left", left_index=True, right_index=True)

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
        f"{rootpath}/plots/prepared_models/{building_name.lower()}_{season}_lstm.html"
    )

    model.summary()
    model.save(
        f"{rootpath}/models_saved/{building_name.lower()}{tower_number}_{season}_lstm/"
    )
