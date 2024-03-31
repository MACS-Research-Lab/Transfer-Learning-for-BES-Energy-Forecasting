import torch.utils.data as data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from typing import List
import model_prep as model_prep  # Assuming you have a module named model_prep which has the relevant functions


class GetLoader(data.Dataset):
    def __init__(
        self,
        building_name: str,
        tower_number: int,
        features: List[str],
        target: str,
        season: str = None,
        train_percentage: float = 0.75,
        use_delta: bool = True,
        shuffle_seed: int = 42,
        step_back: int = 6,  # Add step_back argument if needed
    ):
        """
        This init function will preprocess data based on the create_base_model
        """

        lstm_df, first_val = model_prep.create_preprocessed_lstm_df(
            building_name=building_name,
            tower_number=tower_number,
            features=features,
            target=target,
            season=season,
            use_delta=use_delta,
        )

        X = lstm_df.drop(f"{target}(t)", axis=1)  # drop target column
        y = lstm_df[f"{target}(t)"]  # only have target column

        X_train, _, y_train, _ = train_test_split(
            X,
            y,
            test_size=(1 - train_percentage),
            shuffle=False,
            random_state=shuffle_seed,
        )

        scaler = MinMaxScaler().fit(X_train)
        X_train[X_train.columns] = scaler.transform(X_train)

        self.X_data = model_prep.df_to_3d(
            lstm_dtframe=X_train, num_columns=len(features) + 1, step_back=step_back
        )
        self.y_data = y_train.values

        print("shapes in dataloader (X, y):", self.X_data.shape, self.y_data.shape)

        self.first_val = first_val

    def __getitem__(self, item):
        """
        This method will return data based on the index (item)
        """
        return self.X_data[item], self.y_data[item]

    def __len__(self):
        return len(self.X_data)
