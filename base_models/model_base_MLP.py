# Load Packages
import pandas as pd
import numpy as np
import plotly.express as px

import warnings
from typing import List
import os, sys, time

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense

rootpath = ".."
sys.path.insert(0, f"{os.getcwd()}/{rootpath}/base_models")
sys.path.insert(0, f"{os.getcwd()}/{rootpath}/source_models")
warnings.filterwarnings("ignore")

import model_prep

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
    train_percentage: float = 0.8,
    is_source: bool = False,
    use_delta: bool = False,
    shuffle_seed: int = 42,
):
    """
    1. Load data and do basic preprocessing
    """
    # load data
    df = pd.read_csv(
        f"{rootpath}/data/{building_name.lower()}/{building_name.lower()}{tower_number}_preprocessed.csv",
        index_col="time",
    )
    df.index = pd.to_datetime(df.index)

    # only take data for one season
    df = model_prep.choose_season(df, season=season)

    # remove cases in which tower was OFF, and cases where OFF data would be included in past timesteps of ON data
    on_condition = df[target] > 0
    df = df.drop(df[~on_condition].index, axis=0)

    # select features and targets and create final dataframe that includes only relevant features and targets
    df = df[features + ["DayOfWeek"]].join(df[target], on=df.index)

    # if difference from first value should be used as for predictions then return the first value
    first_val = df.iloc[0, df.columns.get_loc(target)]
    if use_delta:
        df[target] = df[target] - first_val

    """
    2+3. Seasonality removal and train/test split
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df[features + ["DayOfWeek"]],
        df[target],
        test_size=(1 - train_percentage),
        shuffle=True,
        random_state=shuffle_seed,
    )
    train_set = X_train.merge(y_train, how="left", left_index=True, right_index=True)
    test_set = X_test.merge(y_test, how="left", left_index=True, right_index=True)

    # apply seasonality removal
    sdf = calculate_seasonal_index(train_set, target, "DayOfWeek", 7)
    train_set[target] = operate_with_sp(
        col=train_set[target], sp_df=sdf, operation="divide"
    )

    # Split further
    X_train, X_test = train_set[features], test_set[features]
    y_train, y_test = train_set[target], test_set[target]

    # scale feature data - use source domain scaler
    scaler = MinMaxScaler().fit(X_train)
    X_train[X_train.columns] = scaler.transform(X_train)
    X_test[X_test.columns] = scaler.transform(X_test)

    vec_X_train = X_train.values
    vec_X_test = X_test.values
    vec_y_train = y_train.values
    vec_y_test = y_test.values

    """
    4. Create and Train model
    """

    # Build the MLP model - hyperparams are optimized
    model = Sequential()
    model.add(
        Dense(
            80,
            input_shape=(len(features),),
            kernel_initializer="normal",
            activation="relu",
        )
    )
    model.add(Dense(units=1, activation="linear"))

    # Compile the model
    model.compile(optimizer="adam", loss="mean_squared_error")

    epochs = 100 if is_source else 200

    # Train the model
    start_time = time.time()
    history = model.fit(
        vec_X_train,
        vec_y_train,
        epochs=epochs,
        batch_size=10,
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
    # if the model is a source model for a transfer, then it is trained at a different level of epochs so should be saved at a different place
    if is_source:
        model.save(
            f"{rootpath}/results/models_saved/source_models/{building_name.lower()}{tower_number}_{season}_mlp/"
        )

    else:
        results_df = pd.DataFrame(
            {
                "actual": vec_y_test.reshape((vec_y_test.shape[0])),
                "predicted": yhat.reshape((yhat.shape[0])),
            },
            index=y_test.index,
        )
        # invert seasonality removal
        if not use_delta:
            first_val = 0
        results_df["actual"] = (
            operate_with_sp(results_df["actual"], sdf, "multiply") + first_val
        )
        results_df["predicted"] = (
            operate_with_sp(results_df["predicted"], sdf, "multiply") + first_val
        )

        # SAVE ERROR AND DATA AVAILABILITY INFORMATION

        # calculate, display and save error results
        mae = mean_absolute_error(results_df["actual"], results_df["predicted"])
        print(mae)
        rmse = np.sqrt(
            mean_squared_error(results_df["actual"], results_df["predicted"])
        )
        mae_sd = np.std(np.abs(results_df["actual"] - results_df["predicted"]))
        model_prep.save_base_errors(
            "adjustedMLP",
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
                start=results_df.index.min(),
                end=results_df.index.max(),
                freq="5min",
            )
        ).merge(results_df, how="left", left_index=True, right_index=True)

        # display trend of temperature predictions (actual vs predicted)
        fig = px.line(display_df, x=display_df.index, y=["actual", "predicted"])
        fig.update_layout(
            title=f"{building_name} Tower {tower_number} MLP Model Results",
            xaxis_title="time",
            yaxis_title=target,
        )
        fig.show()
        fig.write_html(
            f"{rootpath}/results/plots/prepared_models/{building_name.lower()}_{season}_mlp.html"
        )

        # save the model
        model.save(
            f"{rootpath}/results/models_saved/base_models/{building_name.lower()}{tower_number}_{season}_mlp/"
        )


def calculate_seasonal_index(time_series, target_col_name, seasonality_column, m):
    """
    Calculate the seasonal index for each seasonality value in the time series.

    Parameters:
    - time_series: Pandas DataFrame containing the time series data with a column for the seasonality values.
    - seasonality_column: String representing the column name containing the seasonality values (e.g., days of the week).
    - m: Integer representing the number of data points for each seasonality value.

    Returns:
    - Pandas DataFrame containing the seasonal index for each seasonality value.
    """

    # Group the data by the seasonality column
    grouped_data = time_series.groupby(seasonality_column)

    # Calculate the average of all target variable data points
    y_bar = time_series.mean()[target_col_name]

    # Initialize an empty dictionary to store the seasonal index values
    seasonal_index_dict = {}

    # Iterate through each group (seasonality value)
    for group, group_data in grouped_data:
        # Calculate the sum of the first m data points
        sum_y_p_j = group_data.iloc[:m][target_col_name].sum()

        # Calculate the seasonal index using the provided formula
        seasonal_index = 1 / y_bar * (1 / m) * sum_y_p_j

        # Store the seasonal index value in the dictionary
        seasonal_index_dict[group] = seasonal_index

    # Convert the dictionary to a Pandas DataFrame
    seasonal_index_df = pd.DataFrame(
        list(seasonal_index_dict.items()), columns=[seasonality_column, "sp"]
    )

    return seasonal_index_df


def operate_with_sp(col, sp_df, operation):
    index_col = col.index
    combined_df = pd.merge(
        col, sp_df, left_on=col.index.dayofweek, right_on="DayOfWeek", how="left"
    ).set_index(index_col)
    if operation == "multiply":
        combined_df[col.name] = combined_df[col.name] * combined_df["sp"]
    elif operation == "divide":
        combined_df[col.name] = combined_df[col.name] / combined_df["sp"]
    else:
        raise ValueError("Invalid operation")
    return combined_df[col.name]
