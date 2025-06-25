import os
import sys
import pandas as pd
from typing import Optional
from functools import reduce
from math import sqrt


def loadData(file_path: str = "data.csv") -> Optional[pd.DataFrame]:
    try:
        data = pd.read_csv(file_path)
        return data
    except OSError:
        print(f"Error: Could not open file '{file_path}'")
        return None
    except pd.errors.ParserError:
        print("Error: File contents are not valid or improperly formatted")
        return None


def mean(ls: list[int | float]) -> float:
    '''Calculate the Mean'''
    return sum(ls) / len(ls)


def median(ls: list[int | float]) -> float:
    '''Calculate the Median'''
    if len(ls) == 1:
        return float(ls[0])
    else:
        s = sorted(ls)
        mid = int(len(s)/2)
        return s[mid] if len(ls) % 2 else (s[mid - 1] + s[mid])/2


def var(ls: list[int | float]) -> float:
    '''Calculate the Population Variance'''
    _mean = mean(ls)
    return mean([(x - _mean) ** 2 for x in ls])


def std(ls: list[int | float]) -> float:
    '''Calculate the Standard Deviation'''
    return sqrt(var(ls))


def quartile(ls: list[int | float]) -> tuple[float, float]:
    '''Calculate Quartile (25% and 75%)'''
    if len(ls) == 1:
        return float(ls[0]), float(ls[0])
    s = sorted(ls)
    mid = len(s) // 2
    m1 = mid + 1 if len(s) % 2 else mid
    return float(median(s[:m1])), float(median(s[mid:]))


def min(ls: list[int | float]) -> float:
    return reduce(lambda acc, val : val if acc > val else acc, ls, ls[0])


def max(ls: list[int | float]) -> float:
    return reduce(lambda acc, val : val if acc < val else acc, ls, ls[0])


def describe(df: pd.DataFrame):
    summary = {
        "Count": [],
        "Mean": [],
        "Std": [],
        "Min": [],
        "25%": [],
        "50%": [],
        "75%": [],
        "Max": []
    }

    numeric_columns = df.select_dtypes(include=["number"]).columns

    for col in numeric_columns:
        col_data = df[col].dropna().tolist()
        summary["Count"].append(float(len(col_data)))
        summary["Mean"].append(mean(col_data))
        summary["Std"].append(std(col_data))
        summary["Min"].append(min(col_data))
        q1, q3 = quartile(col_data)
        summary["25%"].append(q1)
        summary["50%"].append(median(col_data))
        summary["75%"].append(q3)
        summary["Max"].append(max(col_data))

    # .T transposes the dataFrame
    summary_df = pd.DataFrame(summary, index=numeric_columns).T

    return summary_df.to_string(float_format="%.6f")


if __name__ == "__main__":
    if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
        print("Error: Please provide a valid file path")
        exit(1)

    try:
        df = loadData(sys.argv[1])
        if df is None:
            raise ValueError("loadData returned None")
        
        # df = df.drop(columns=['Index', 'index'], errors='ignore')

        print(describe(df))

        # print("\n\n")
        # print(df.describe())

    except Exception:
        print("Error: Failed to process or describe the dataset")
        exit(1)
