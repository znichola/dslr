from describe import loadData, trainDataFilePath


def loadWeights(file_path="weights.txt") -> list[float] | None:
    return loadData(file_path)


if __name__ == "__main__":
    print("Let's predict the house for our students!")
    print()

    weights = loadWeights()
    if weights is None:
        exit(1)

    data = loadData(trainDataFilePath())
    if data is None:
        exit(1)

    print(weights)