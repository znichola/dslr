import pandas as pd
import matplotlib.pyplot as plt
import math
from describe import loadData, trainDataFilePath

if __name__ == "__main__":
    data = loadData(trainDataFilePath())
    if data is None:
        exit(1)


    df = data

    print("Which Hogwarts course has a homogeneous score distribution between all four houses?")

    numeric_cols = df.select_dtypes(include='number').columns
    num_plots = len(numeric_cols)
    all_houses = df["Hogwarts House"].dropna().unique()

    cols = math.ceil(math.sqrt(num_plots))
    rows = math.ceil(num_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), squeeze=False)

    for i, column in enumerate(numeric_cols):
        ax = axes[i // cols][i % cols]
        for house in all_houses:
            df[df["Hogwarts House"] == house][column].plot(kind='hist', bins=30, alpha=0.5, label=house, ax=ax)

        ax.set_title(f'"{column}" by House')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()

    plt.tight_layout()
    plt.show()
