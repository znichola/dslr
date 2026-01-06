import pickle
from test import logistic_regression

def main(data):
    foo = logistic_regression("datasets/datasets_test.csv")
    foo._weights = data["weights"]



if __name__ == "__main__":
    with open("weights.pkl", "rb") as f:
        data = pickle.load(f)

        print(data["houses"])

        main(data)

