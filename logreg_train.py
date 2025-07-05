import pandas as pd
from describe import loadData, trainDataFilePath


def saveWeightsToFile(file_path: str, weights: pd.DataFrame) -> None:
    try:
        weights.to_csv(file_path)
        print(f"Weights saved to {file_path}")
    except:
        print(f"Error: Could not save weights to '{file_path}'")


if __name__ == "__main__":
    print("Let's train the model to predict the correct house!")
    print()

    data = loadData(trainDataFilePath())
    if data is None:
        exit(1)

    df = data

    print(df)