import os
import sys
import math
import pandas as pd
from describe import loadData, trainDataFilePath


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
    return 1 / (1 + math.pow(math.e, -z))


# g(θ^T x) the T comes from transpose, so transpose theta and multiply by x
#          like a dot product t0*x0 + t1*x1 ... tn*xn
def hypothesis(theta_vec, x_vec) -> float:
    '''[θ^T * x] == [θ dot product x]'''
    return sigmoid(sum(theta * x for theta, x in zip(theta_vec, x_vec)))


# The cost/loss for of an estimate of whether or not a student belongs to
# the house these weights are trained to predict
def cost(theta_vec, x_vec, y) -> float:
    '''y_i * log(hθ(x_i)) + (1-y_i) * log(1-hθ(x_i))'''

    # x_vec : the subject grades for the student
    # y     : the answer, 1 if yes the student i part of this house, 0 if not

    # the model's estimate for the probability of a student 
    # belongin to the house the weights are trained on
    h = hypothesis(theta_vec, x_vec)

    return (y * math.log(h)) + ((1 - y) * math.log(1 - h))

def cost(h, y) -> float:
    '''y_i * log(hθ(x_i)) + (1-y_i) * log(1-hθ(x_i))'''
    return (y * math.log(h)) + ((1 - y) * math.log(1 - h))


def partial_derivative(h, y, j):
    '''(hθ(x_i) - y_i) * x_i_j'''
    return (h - y) * j


def cleanUpData(df: pd.DataFrame) -> pd.DataFrame:
    drop_classes = ['Arithmancy', 'Defense Against the Dark Arts', 'Transfiguration', 'Care of Magical Creatures', 'Flying']
    unused = ['First Name', 'Last Name', 'Birthday', 'Best Hand']
    columns_to_drop = ['Index'] + drop_classes + unused

    return df.drop(columns=columns_to_drop).dropna() # also drop lines with missing data


def train(df: pd.DataFrame, learning_rate = 0.05, num_iterations = 60):
    # all_houses = df["Hogwarts House"].unique()
    all_houses = ['Ravenclaw', 'Slytherin', 'Gryffindor', 'Hufflepuff'] # for type hints
    feature_cols = df.select_dtypes(include='number').columns
    num_features = len(feature_cols)

    num_students = len(df)

    weights_per_house = pd.DataFrame(
        data=0,
        index=feature_cols,
        columns=all_houses
    )

    ans_per_house = pd.get_dummies(df["Hogwarts House"])
    df = df.drop(columns=["Hogwarts House"])


    for generation in range(num_iterations):
        for house in all_houses:
            w_vec = weights_per_house[house].values
            y_vec = ans_per_house[house].values

            h_vec = df.apply(lambda x : hypothesis(w_vec, x), axis="columns")
            # loss = [cost(h, y) for h, y in zip(h_vec, y_vec)]
 
            h_y_diff = h_vec - y_vec

            gradients = pd.DataFrame(data=0.0, index=feature_cols, columns=[0])

            for feature in feature_cols:
                x_j = df[feature].values
                gradients.loc[feature, 0] = (h_y_diff * x_j).mean()

            weights_per_house[house] = weights_per_house[house] - learning_rate * gradients[0]
            # print(weights_per_house)

    return weights_per_house



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

    df = cleanUpData(data)

    weights = train(df)

    print(weights)

    saveWeightsToFile(weights)
