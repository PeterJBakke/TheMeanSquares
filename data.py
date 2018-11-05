"""
Data preprocessing file

Created: 2018-11-05
Author: Peter J. Bakke

Reviewed by:


Changed by:



"""

import pandas as pd
import os



class MovieLens():
    """
    Class to handle the MovieLens data
    """
    def __init__(self):
        self.base_dir = os.getcwd()
        self.ratingsPath = self.base_dir + '/Datasets/MovieLens-Small/ratings.csv'
        self.moviesPath = self.base_dir + '/Datasets/MovieLens-Small/movies.csv'

    def MovieLensRatingsData(self):
        """
        Method for loading ratings data
        :return: data frame with MovieLens ratings data
        """
        path = self.ratingsPath
        data = pd.read_csv(path)
        return data

    def MovieLensMoviesData(self):
        """
        Method for loading movies data
        :return: data frame with MovieLens movies data
        """
        path = self.moviesPath
        data = pd.read_csv(path)
        return data

    def MovieLensUsers(self):
        """
        Method for getting a list of unique userId's
        :return: DataFrame with the userId's from the MovieLens DataSet
        """
        df = self.MovieLensRatingsData()
        df = df['userId']
        df.drop_duplicates( inplace=True)
        return df









if __name__ == "__main__":
    myData = MovieLens()
    movies = myData.MovieLensMoviesData()
    ratings = myData.MovieLensRatingsData()
    users = myData.MovieLensUsers()
    print(movies.shape)
    print(ratings.shape)
    print(users.shape)

