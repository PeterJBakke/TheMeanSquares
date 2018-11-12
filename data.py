"""
Data pre processing file

Created: 2018-11-05
Author: Peter J. Bakke

Reviewed by:


Changed by:



"""

import pandas as pd
import os
from torchtext import data
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


class MovieLens():
    """
    Class to handle the MovieLens data
    """
    def __init__(self):
        self.base_dir = os.getcwd()
        self.ratingsPath = os.path.join(self.base_dir, 'Datasets', 'MovieLens-Small', 'ratings.csv')
        self.moviesPath = os.path.join(self.base_dir, 'Datasets', 'MovieLens-Small', 'movies.csv')
        
        print("MovieLens Data handler started \n")
        print("Path to ratings data: \n" + str(self.ratingsPath))
        print("Path to movies data: \n" + str(self.moviesPath))

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







class TalentFox():
    """
    Class to handle the TalentFox data

    Predict:
    match_status

    Columns for candidate:
    candidate_city, candidate_state, candidate_country, candidate_title, candidate_birth_date,
    candidate_current_fixed_salary, candidate_current_bonus_salary, candidate_in_job_market_since,
    candidate_other_languages, candidate_is_looking_for_new_job, candidate_wish_2, candidate_wish_3, candidate_wish_1,
    candidate_education, candidate_language_negotiative, candidate_language_basic, candidate_language_fluent,
    candidate_highest_degree, candidate_career_type, candidate_industries, candidate_professions, candidate_resume,
    candidate_feedback, candidate_professions_global, candidate_industries_global, candidate_relocation_ready,

    Columns for job:
    job_fixed_salary, job_bonus_salary, job_title, job_vacation_days, job_needed_experience, job_language,
    job_description, job_daily_tasks_of_job, job_required_experience_of_candidate,
    job_preferred_experience_of_candidate, job_preferred_education_of_candidate, job_max_candidate_age,
    job_min_candidate_age, job_company_structure, job_language_skills_negotiative, job_language_skills_basic,
    job_candidate_radius, job_candidate_relocation, job_city, job_state, job_country, job_time_model, job_max_salary,
    job_questions_for_candidate, match_employer_feedback
    """
    def __init__(self):
        self.base_dir = os.getcwd()
        self.file = os.path.join(self.base_dir, 'Datasets/talentfox_match_data/processed_dataset.csv')
        self.data = self.load(self.file)

    def load(self, file):
        data = pd.read_csv(file)
        return data

    def getJobRequiredExperience(self):
        return self.data['job_required_experience_of_candidate']

    def getJobDescription(self):
        return self.data['job_description']

    def getJobDailyTasks(self):
        return self.data['job_daily_tasks_of_job']

    def getJobTitles(self):
        return self.data['job_title']

    def getCandidateResume(self):
        return self.data['candidate_resume']

    def getCandidateProfessions(self):
        return self.data['candidate_professions']

    def getCandidateTitle(self):
        return self.data['candidate_title']

    def getMatchStatus(self):
        return self.data['match_status']


class citeulike():
    """
    Class to handle the Cite-U-Like data

    Predict:
    match_status

    """
    def __init__(self):
        self.base_dir = os.getcwd()
        self.file_user_info = os.path.join(self.base_dir, 'Datasets/citeulike/user-info.csv')
        self.file_raw_data = os.path.join(self.base_dir, 'Datasets/citeulike/raw-data.csv')
        self.user_info = self.load(self.file_user_info)
        self.raw_data = self.load(self.file_raw_data)

    def load(self, file):
        data = pd.read_csv(file, encoding="ISO-8859-1")
        return data

    def getUserInfo(self):
        return self.user_info

    def getRawData(self):
        return self.raw_data


if __name__ == "__main__":
    myData = MovieLens()
    movies = myData.MovieLensMoviesData()
    ratings = myData.MovieLensRatingsData()
    users = myData.MovieLensUsers()
    print(movies.shape)
    print(ratings.shape)
    print(users.shape)






"""
Notes from lecture 2018-11-12 on Data Loaders

Vocabulary 
 -- can be implemented with TorchText

Padding - dynamic batching
 -- You don't want information from the padding to affect your computation
 -- Can also be implemented with TorchText
 -- PyTorch.LSTM has a method that takes length as input that also ensures that we don't calculate on the padding


The dataloader should be a function that acts as an iterator that only gives one batch at a time

- train_iter (only handles one batch at a time during training)
- valid_iter (only handles one batch at a time during validation)
- test_iter (only handles one batch at a time during testing)
 
"""