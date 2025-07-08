import os
import sys
from describe import loadData, trainDataFilePath


def loadWeights(file_path="weights.csv") -> list[float] | None:
    return loadData(file_path)


if __name__ == "__main__":
    print("Let's predict the house for our students!")
    print()

    # if len(sys.argv) < 3 or not os.path.isfile(sys.argv[1]) or not os.path.isfile(sys.argv[2]):
    #     print("Error: Please provide a valid file pathsm for data.csv and weights.csv")
    #     exit(1)
    
    # data_path = sys.argv[1]
    # weights_path = sys.argv[2]


    # TODO : switch to using the args
    weights = loadWeights()
    if weights is None:
        exit(1)

    # TODO : switch to using arg for the input
    data = loadData(trainDataFilePath())
    if data is None:
        exit(1)

    print(weights)