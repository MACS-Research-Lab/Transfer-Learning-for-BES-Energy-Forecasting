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
import keras

rootpath = ".."
sys.path.insert(0, f"{os.getcwd()}/{rootpath}/base_models")
sys.path.insert(0, f"{os.getcwd()}/{rootpath}/source_models")
warnings.filterwarnings("ignore")

import model_prep


step_back = 6  # window size = 6*5 = 30 mins


def transfer_autoLSTM(
    from_building_name: str,
    from_tower_number: int,
    to_building_name: str,
    to_tower_number: int,
    to_features: List[str],
    to_target: str,
    to_season: str = None,
    from_season: str = None,
    finetuning_percentage: float = 0,
    finetune_epochs: int = 100,
    display_results: bool = True,
    use_delta: bool = False,
    shuffle_seed: int = 42,
):
    # fix inputs
    if from_season == None and to_season != None:
        from_season = to_season

    training_time = 0

    """
    1. Load data and do LSTM preprocessing
    """

    lstm_to_df, to_first_val = model_prep.create_preprocessed_lstm_df(
        building_name=to_building_name,
        tower_number=to_tower_number,
        features=to_features,
        target=to_target,
        season=to_season,
        use_delta=use_delta,
    )
    if not to_season:
        to_season = from_season = "allyear"

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

        # scale feature data
        X_test[X_test.columns] = MinMaxScaler().fit_transform(X_test)

        # create 3d vector form of data
        vec_X_test = model_prep.df_to_3d(
            lstm_dtframe=X_test, num_columns=len(to_features) + 1, step_back=step_back
        )
        vec_y_test = y_test.values
        # print(vec_X_test.shape, vec_y_test.shape)

        # load model
        model = keras.models.load_model(
            f"../results/models_saved/base_models/{from_building_name.lower()}{from_tower_number}_{from_season}_eld/"
        )

    # if finetuning is required
    else:
        # split train and test set
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=(1 - finetuning_percentage),
            shuffle=True,
            random_state=shuffle_seed,
        )

        # scale feature data
        scaler = MinMaxScaler()
        scaler = scaler.fit(X_train)
        X_train[X_train.columns] = scaler.transform(X_train)
        X_test[X_test.columns] = scaler.transform(X_test)

        # create 3d vector form of data
        vec_X_train = model_prep.df_to_3d(
            lstm_dtframe=X_train,
            num_columns=len(to_features) + 1,
            step_back=step_back,
        )
        vec_X_test = model_prep.df_to_3d(
            lstm_dtframe=X_test,
            num_columns=len(to_features) + 1,
            step_back=step_back,
        )

        vec_y_train = y_train.values
        vec_y_test = y_test.values

        # if model finetuning has already been done, simply load the model
        model_path = f"{rootpath}/results/models_saved/eld_ft/{from_building_name.lower()}{from_tower_number}{from_season}_to_{to_building_name.lower()}{to_tower_number}{to_season}_ft{int(finetuning_percentage*100)}_seed{shuffle_seed}/"

        if os.path.exists(model_path):
            model = keras.models.load_model(model_path)
            print(f"Pre-saved model for ft={finetuning_percentage} seed={shuffle_seed}")

        # if model finetuning has not been done, finetune a base model
        else:
            # load and finetune model
            model = keras.models.load_model(
                f"{rootpath}/results/models_saved/source_models/{from_building_name.lower()}{from_tower_number}_{from_season}_eld/"
            )
            print(f"Finetuning for ft={finetuning_percentage} seed={shuffle_seed}")

            # freeze encoder and decoder layers
            model.layers[1].trainable = False
            model.layers[3].trainable = False
            model.layers[4].trainable = False
            # keep lstm layer trainable
            model.layers[0].trainable = True
            model.layers[2].trainable = True

            model.compile(
                optimizer=keras.optimizers.Adam(1e-5),  # Very low learning rate
                loss="mse",
                metrics=[keras.metrics.BinaryAccuracy()],
            )
            start_time = time.time()
            model.fit(
                vec_X_train,
                vec_y_train,
                epochs=finetune_epochs,
                verbose=0,
                shuffle=False,
            )
            end_time = time.time()
            training_time = end_time - start_time

            model.save(model_path)

    """
    3. Predict
    """

    yhat = model.predict(vec_X_test)

    """
    4. Display results
    """

    # show results
    results_df = pd.DataFrame(
        {
            "actual": vec_y_test.reshape((vec_y_test.shape[0])),
            "predicted": yhat.reshape((yhat.shape[0])),
        },
        index=y_test.index,
    )

    if use_delta:
        results_df["actual"] = results_df["actual"] + to_first_val
        results_df["predicted"] = results_df["predicted"] + to_first_val

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
            title=f"{from_building_name} Tower {from_tower_number} {from_season} model used on {to_building_name} Tower {to_tower_number} {to_season} ({finetuning_percentage*100}% fine-tuning) LSTM-Autoencoder Model Results",
            xaxis_title="time",
            yaxis_title=to_target,
        )
        return fig

    if display_results:
        fig = display_transfer_results()
    else:
        fig = None
    return rmse, fig, mabs_error, training_time, len(X)
