import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
from describe import loadData, trainDataFilePath

if __name__ == "__main__":
    data = loadData(trainDataFilePath())
    if data is None:
        exit(1)


    df = data

    print("What are the two features that are similar ?")

    numeric_cols = df.select_dtypes(include='number').columns
    all_houses = df["Hogwarts House"].dropna().unique()

    feature_pairs = list(itertools.combinations(numeric_cols, 2))

    num_plots = len(feature_pairs)
    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(cols, rows, figsize=(cols*2, rows*2), squeeze=True)

    i = 0
    for x_feature, y_feature in feature_pairs:
        ax = axes[i // cols][i % cols]
        i += 1
        for house in all_houses:
            house_bool_maks = df["Hogwarts House"] == house
            ax.scatter(df[house_bool_maks][x_feature],
                        df[house_bool_maks][y_feature],
                        s=10, alpha=0.5)

    plt.tight_layout()
    plt.show()
