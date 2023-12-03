# Load Packages
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px

import warnings
import os, sys
from typing import List
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import keras

rootpath = ".."
sys.path.insert(0, f"{os.getcwd()}/{rootpath}/base_models")
import model_prep
import model_base_MLP

warnings.filterwarnings("ignore")


step_back = 6  # window size = 6*5 = 30 mins
season_map = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "fall": [9, 10, 11],
    "winter": [12, 1, 2],
}


def transfer_adjustedMLP(
    from_building_name: str,
    from_tower_number: int,
    to_building_name: str,
    to_tower_number: int,
    features: List[str],
    target: str,
    to_season: str = None,
    from_season: str = None,
    finetuning_percentage: float = 0,
    finetune_epochs: int = 100,
    display_results: bool = True,
    use_delta: bool = True,
    shuffle_seed: int = 42,
):
    """
    1. Load data and do basic preprocessing
    """
    # load data
    df = pd.read_csv(
        f"{rootpath}/data/{to_building_name.lower()}/{to_building_name.lower()}{to_tower_number}_preprocessed.csv",
        index_col="time",
    )
    df.index = pd.to_datetime(df.index)

    # only take data for one season
    df = model_prep.choose_season(df, season=to_season)

    # remove cases in which tower was OFF, and cases where OFF data would be included in past timesteps of ON data
    on_condition = df[target] > 0
    df = df.drop(df[~on_condition].index, axis=0)

    # select features and targets and create final dataframe that includes only relevant features and targets
    df = df[features + ["DayOfWeek"]].join(df[target], on=df.index)

    # if difference from first value should be used as for predictions then return the first value
    first_val = df.iloc[0, df.columns.get_loc(target)] if use_delta else 0
    df[target] = df[target] - first_val

    """
    2,3. Split data into training and testing sets + Seasonality removal
    """

    X_train, X_test, y_train, y_test = train_test_split(
        df[features + ["DayOfWeek"]],
        df[target],
        test_size=(1 - finetuning_percentage) if finetuning_percentage != 0 else 0.8,
        shuffle=True,
        random_state=shuffle_seed,
    )
    train_set = X_train.merge(y_train, how="left", left_index=True, right_index=True)
    test_set = X_test.merge(y_test, how="left", left_index=True, right_index=True)

    # find seasonality removal using train set
    sdf = model_base_MLP.calculate_seasonal_index(train_set, target, "DayOfWeek", 7)
    train_set[target] = model_base_MLP.operate_with_sp(
        col=train_set[target], sp_df=sdf, operation="divide"
    )

    # Split further
    X_train, X_test = train_set[features], test_set[features]
    y_train, y_test = train_set[target], test_set[target]

    # scale feature data
    scaler = MinMaxScaler().fit(X_train)
    X_train[X_train.columns] = scaler.transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)
    vec_X_train = X_train.values
    vec_X_test = X_test.values

    vec_y_train = y_train.values
    vec_y_test = y_test.values

    training_time = 0

    # if no finetuning is required
    if finetuning_percentage == 0:
        # load base model
        model = keras.models.load_model(
            f"{rootpath}/results/models_saved/base_models/{from_building_name.lower()}{from_tower_number}_{from_season}_mlp/"
        )

    else:
        # load source model
        model = keras.models.load_model(
            f"{rootpath}/results/models_saved/source_models/{from_building_name.lower()}{from_tower_number}_{from_season}_mlp/"
        )
        # finetune
        start_time = time.time()
        model.trainable = True

        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
            loss="mse",
            metrics=[keras.metrics.BinaryAccuracy()],
        )
        history = model.fit(
            vec_X_train, vec_y_train, epochs=finetune_epochs, batch_size=10, verbose=0
        )
        end_time = time.time()
        training_time = end_time - start_time

        model.save(
            f"{rootpath}/results/models_saved/mlp_ft/{from_building_name.lower()}{from_tower_number}{from_season}_to_{to_building_name.lower()}{to_tower_number}{to_season}_ft{int(finetuning_percentage*100)}/"
        )

    # Evaluate the model
    y_pred = model.predict(vec_X_test)
    results_df = pd.DataFrame(
        {
            "actual": vec_y_test.reshape((vec_y_test.shape[0])),
            "predicted": y_pred.reshape((y_pred.shape[0])),
        },
        index=y_test.index,
    )

    if not use_delta:
        first_val = 0
    # invert seasonality removal
    results_df["actual"] = (
        model_base_MLP.operate_with_sp(results_df["actual"], sdf, "multiply")
        + first_val
    )
    results_df["predicted"] = (
        model_base_MLP.operate_with_sp(results_df["predicted"], sdf, "multiply")
        + first_val
    )

    rmse = np.sqrt(mean_squared_error(results_df["actual"], results_df["predicted"]))
    mabs_error = mean_absolute_error(results_df["actual"], results_df["predicted"])

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
            title=f"{from_building_name} Tower {from_tower_number} {from_season} model used on {to_building_name} Tower {to_tower_number} {to_season} ({finetuning_percentage*100}% fine-tuning) MLP Model Results",
            xaxis_title="time",
            yaxis_title=target,
        )
        return fig

    if display_results:
        fig = display_transfer_results()
    else:
        fig = None

    return rmse, fig, mabs_error, training_time, len(df)
