import pandas as pd
from typing import Optional
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


def median(ls: list[int | float]) -> int | float:
    '''Calculate the Median'''
    if len(ls) == 1:
        return ls[0]
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


def quartile(ls: list[int | float]) -> list[int | float]:
    '''Calculate Quartile (25% and 75%)'''
    if len(ls) == 1:
        return [ls[0], ls[0]]
    s = sorted(ls)
    mid = int(len(s)/2)
    m1 = mid + 1 if len(s) % 2 else mid
    return [float(median(s[:m1])), float(median(s[mid:]))]


def describe(file_path: str = "data.csv"):
    data = loadData(file_path)
    if data is None:
        exit(1)
    

if __name__ == "__main__":
    main()
