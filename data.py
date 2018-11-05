"""
Data preprocessing file
"""

import pandas as pd
import os

base_dir = os.getcwd()

def MovieLensRatingsData(path=base_dir + '/Datasets/MovieLens-Small/ratings.csv'):
    """
    Data loader for the Movie Lens Ratings data
    :param path: Directory where data is located
    :return: pandas dataframe with MovieLens Ratings data
    """
    data = pd.read_csv(path)
    return data

if __name__ == "__main__":
    myData = MovieLensRatingsData()
    print(myData)

