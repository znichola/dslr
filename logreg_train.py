import os
import sys
import math
import pandas as pd
from describe import loadData, trainDataFilePath, mean
import matplotlib.pyplot as plt
import numpy as np

def saveWeightsToFile(weights: pd.DataFrame, file_path: str = "weights.csv") -> None:
    try:
        weights.to_csv(file_path)
        print(f"Weights saved to {file_path}")
    except:
        print(f"Error: Could not save weights to '{file_path}'")


# Sigmoid function that places all inputs between 0 and 1
def sigmoid(z) -> float:
    '''g(z) = 1 / (1 + e-z)'''
    z = max(min(z, 100), -100)  # clamp z to [-100, 100]
    return 1 / (1 + math.exp(-z))


# g(θ^T x) the T comes from transpose, so transpose theta and multiply by x
#          like a dot product t0*x0 + t1*x1 ... tn*xn
def hypothesis(theta_vec, x_vec) -> float:
    '''[θ^T * x] == [θ dot product x]'''
    return sigmoid(sum(theta * x for theta, x in zip(theta_vec, x_vec)))


# The cost/loss for of an estimate of whether or not a student belongs to
# the house these weights are trained to predict
# def cost(theta_vec, x_vec, y) -> float:
#     '''y_i * log(hθ(x_i)) + (1-y_i) * log(1-hθ(x_i))'''

#     # x_vec : the subject grades for the student
#     # y     : the answer, 1 if yes the student i part of this house, 0 if not

#     # the model's estimate for the probability of a student 
#     # belongin to the house the weights are trained on
#     h = hypothesis(theta_vec, x_vec)
#     if h <= 0 or h >= 1:
#         print(f"Suspicious h: {h}, y: {y}")
#     epsilon = 1e-15
#     h = max(min(h, 1 - epsilon), epsilon)
#     return - (y * math.log(h)) + ((1 - y) * math.log(1 - h))

def cost(h, y) -> float:
    '''y_i * log(hθ(x_i)) + (1-y_i) * log(1-hθ(x_i))'''
    if h <= 0 or h >= 1:
        print(f"Suspicious h: {h}, y: {y}")
    epsilon = 1e-15
    h = max(min(h, 1 - epsilon), epsilon)
    return - (y * math.log(h)) + ((1 - y) * math.log(1 - h))


def partial_derivative(h, y, j):
    '''(hθ(x_i) - y_i) * x_i_j'''
    return (h - y) * j


def cleanUpData(df: pd.DataFrame) -> pd.DataFrame:
    drop_classes = ['Arithmancy', 'Defense Against the Dark Arts', 'Transfiguration', 'Care of Magical Creatures', 'Flying']
    unused = ['First Name', 'Last Name', 'Birthday', 'Best Hand']
    columns_to_drop = ['Index'] + drop_classes + unused

    return df.drop(columns=columns_to_drop).dropna() # also drop lines with missing data


def normalizeData(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    feature_mean = df.mean(numeric_only=True)
    feature_std = df.std(numeric_only=True)
    numerical_features = df.select_dtypes('number').columns
    normalized_dataset = (
        df[numerical_features] - feature_mean
    ) / feature_std

    normalized_dataset['Hogwarts House'] = df['Hogwarts House']
    return normalized_dataset, feature_mean, feature_std


def train(df: pd.DataFrame, learning_rate = 0.1, num_iterations = 100):
    # all_houses = df["Hogwarts House"].unique()
    all_houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'] # for type hints
    feature_cols = df.select_dtypes(include='number').columns
    num_features = len(feature_cols)
    num_students = len(df)

    weights_per_house = pd.DataFrame(
        data=0.0,
        index=feature_cols,
        columns=all_houses
    )

    ans_per_house = pd.get_dummies(df["Hogwarts House"])
    df = df.drop(columns=["Hogwarts House"])

    loss_history = {house: [] for house in all_houses}

    X = df.values  # shape: (m, n)
    Y = ans_per_house.values  # shape: (m, k)

    for iteration in range(num_iterations):
        for house_idx, house in enumerate(all_houses):
            theta = weights_per_house[house].values  # shape: (n,)
            y = Y[:, house_idx]  # shape: (m,)

            # Compute predictions for all students
            z = X @ theta  # shape: (m,)
            h = 1 / (1 + np.exp(-np.clip(z, -100, 100)))  # sigmoid vectorized

            # Compute loss
            epsilon = 1e-15
            h = np.clip(h, epsilon, 1 - epsilon)
            loss = - (y * np.log(h) + (1 - y) * np.log(1 - h))
            loss_history[house].append(np.mean(loss))

            # Compute gradient
            gradient = (1 / num_students) * (X.T @ (h - y))  # shape: (n,)

            # Update weights
            weights_per_house[house] = theta - learning_rate * gradient

    return weights_per_house, pd.DataFrame(loss_history)




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

    df, feature_mean, feature_std = normalizeData(cleanUpData(data))

    df.insert(0, 'Bias', 1.0)

    print(df)

    weights, loss = train(df)

    feature_mean['Bias'] = 0.0
    feature_std['Bias'] = 1.0

    weights['Mean'] = feature_mean.values
    weights['Std'] = feature_std.values
    
    print(weights)

    saveWeightsToFile(weights)

    loss.plot(figsize=(10, 6), title="Training Loss per House")
    plt.xlabel("Iteration")
    plt.ylabel("Average Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
