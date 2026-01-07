from logistic_regression import logistic_regression
import pandas as pd
import pickle
import argparse

def loadModel(file_path: str = "model.pkl"):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            print(f"Model loaded from {file_path}")
            return data
    except:
        print(f"Error: Could not load model from '{file_path}'")
        return None


def savePredictionsToFile(predictions, file_path: str = "houses.csv"):
    try:
        df = pd.DataFrame(predictions, columns=["Hogwarts House"])
        df.index.name = "Index"
        df.to_csv(file_path)
        print(f"House predictions saved to {file_path}")
    except:
        print(f"Error: Could not save house predictions to '{file_path}'")


def setupArgs():
    parser = argparse.ArgumentParser(description=("House prediction\n"))
    parser.add_argument(
        "dataset_path",
        type=str
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="model.pkl"
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = setupArgs()

        model = loadModel(args.model_path)
        if model is None:
            exit(1)

        lr = logistic_regression(args.dataset_path, model)

        predictions = lr.predict()

        savePredictionsToFile(predictions)
    except Exception as error:
        print(f"Error: {error}")
        exit(1)


