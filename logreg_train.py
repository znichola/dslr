import pandas as pd
from describe import loadData, trainDataFilePath


def saveWeightsToFile(weights: pd.DataFrame, file_path: str = "weights.csv") -> None:
    try:
        weights.to_csv(file_path)
        print(f"Weights saved to {file_path}")
    except:
        print(f"Error: Could not save weights to '{file_path}'")


if __name__ == "__main__":
    print("Let's train the model to predict the correct house!")
    print()

    # if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
    #     print("Error: Please provide a valid file path for data")
    #     exit(1)

    # data_path = sys.argv[1]

    # TODO : switch to using args for path
    data = loadData(trainDataFilePath())
    if data is None:
        exit(1)

    df = data

    print(df)