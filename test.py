import os
import sys
import math
import pandas as pd
from describe import loadData, trainDataFilePath, mean, describe
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":

    df = loadData(trainDataFilePath())
    if df is None:
        exit(1)

    subjects = df.select_dtypes(include=["number"]).columns

    df_copy = df.copy()

    tmp, _ = describe(df)

    for course in subjects:
        xmin = tmp[course]["Min"]
        xrange = tmp[course]["Range"]
        if xrange == 0:
            df_copy[course] = [1] * len(df_copy[course])
        else:
            df_copy[course] = [(x - xmin) / xrange for x in df_copy[course]]

    print(df_copy)
