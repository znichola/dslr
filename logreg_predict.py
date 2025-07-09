import os
import sys
import pandas as pd
from describe import loadData, trainDataFilePath
from logreg_train import hypothesis, cleanUpData, saveWeightsToFile
import matplotlib.pyplot as plt


def loadWeights(file_path="weights.csv") -> pd.DataFrame | None:
    return loadData(file_path)


def normalizeDataWithStats(df: pd.DataFrame, feature_mean: pd.Series, feature_std: pd.Series) -> pd.DataFrame:
    numerical_features = df.select_dtypes('number').columns
    normalized_dataset = (df[numerical_features] - feature_mean) / feature_std
    normalized_dataset['Hogwarts House'] = df['Hogwarts House']
    return normalized_dataset


def predict_houses(df: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    all_houses = df["Hogwarts House"].unique()

    feature_cols = df.select_dtypes(include='number').columns
    predictions = []

    print(feature_cols)

    for i, student in df.iterrows():
        scores = {}
        for house in all_houses:

            weight_vec = weights[house].values
            x_vec = student[feature_cols].values
           
            score = hypothesis(weight_vec, x_vec)
            
            scores[house] = score
        predicted_house = max(scores, key=scores.get)
        predictions.append(predicted_house)
    return pd.DataFrame(predictions)


if __name__ == "__main__":
    print("Let's predict the house for our students!")
    print()

    # if len(sys.argv) < 3 or not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
    #     print("Error: Please provide a valid file pathsm for data.csv and weights.csv")
    #     exit(1)
    
    # data_path = sys.argv[1]
    # weights_path = sys.argv[2]


    # TODO : switch to using the args
    raw_weights = loadWeights()
    if raw_weights is None:
        exit(1)

    # TODO : switch to using arg for the input
    data = loadData(trainDataFilePath())
    if data is None:
        exit(1)
    
    data = cleanUpData(data)

    weights = raw_weights.drop(columns=["Mean", "Std"])
    feature_mean = raw_weights["Mean"]
    feature_std = raw_weights["Std"]

    predictions = predict_houses(data, weights)
    saveWeightsToFile(predictions, "houses.csv")

    print(predictions)

    true_labels = data["Hogwarts House"].reset_index(drop=True)
    predicted_labels = predictions[0].reset_index(drop=True)

    correct = (true_labels == predicted_labels)
    num_correct = correct.sum()
    num_total = len(correct)
    num_incorrect = num_total - num_correct
    accuracy = num_correct / num_total

    print(f"Accuracy: {accuracy * 100:.2f}%")