import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def show_seasonal_boxplots(
    datadf: pd.DataFrame, column_names: List[str], season_col_name: str = "season"
):
    datadf[season_col_name] = datadf.index.month.map(
        lambda x: "spring"
        if x in [1, 2, 3]
        else "summer"
        if x in [4, 5, 6]
        else "fall"
        if x in [7, 8, 9]
        else "winter"
    )

    # generate separate boxplots for each column
    fig, ax = plt.subplots(len(column_names))
    fig.set_size_inches((12, 4 * len(column_names)))
    for i, column_name in enumerate(column_names):
        sns.boxplot(x="season", y=column_name, data=datadf, ax=ax[i])

    datadf.drop("season", axis=1, inplace=True)
    plt.show()
