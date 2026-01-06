from logistic_regression import logistic_regression
import pandas as pd
import pickle


def loadModel(file_path: str = "model.pkl"):
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
            print(f"Model loaded from {file_path}")
            return data
    except:
        print(f"Error: Could not load model from '{file_path}'")
        return None


def savePredictionsToFile(predictions, file_path: str = "predictions.csv"):
    try:
        predictions.to_csv(file_path)
        print(f"Predictions saved to {file_path}")
    except:
        print(f"Error: Could not save predictions to '{file_path}'")


if __name__ == "__main__":
    data = loadModel()
    if data is None:
        exit(1)

    lr = logistic_regression("datasets/dataset_train.csv")
    lr._weights = data["weights"]
    lr._normalize = data["normalize"]
    lr._houses = data["houses"]
    predictions = lr.predict()
    savePredictionsToFile(pd.DataFrame(predictions))
    foo = list(zip(lr._houses_data, predictions))
    [print(a, "!=", b) if a != b else 0 for a, b in foo]


