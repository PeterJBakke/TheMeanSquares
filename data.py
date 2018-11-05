"""
Data preprocessing file
"""

import pandas as pd
import os

base_dir = os.getcwd()

def MovieLensRatingsData(path=base_dir + '/Datasets/MovieLens-Small/ratings.csv'):
    data = pd.read_csv(path)
    return data

if __name__ == "__main__":
    myData = MovieLensRatingsData()
    print(myData)

