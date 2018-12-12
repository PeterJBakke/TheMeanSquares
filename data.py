"""
Data pre-processing file
"""

import pandas as pd
import numpy as np
import os
from torchtext import data, vocab
import torch
import spacy
from random import randint

spacy_en = spacy.load('en')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MovieLens2:
    """
    New version of MovieLens data handler
    """

    def __init__(self, batch_size=100):
        self.user = data.Field(sequential=False, use_vocab=True, unk_token=None)
        self.movie = data.Field(sequential=False, use_vocab=True)
        self.rating = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

        self.train_set, self.validation_set, self.test_set = data.TabularDataset.splits(
            path='./Datasets/MovieLens-Small/',
            train='train_data.csv',
            validation='val_data.csv',
            test='test_data.csv',
            format='csv',
            fields=[
                ('user', self.user),
                ('movie', self.movie),
                ('rating', self.rating),
            ],
            skip_header=True
        )

        self.train_iter, self.validation_iter, self.test_iter = data.Iterator.splits(
            (self.train_set, self.validation_set, self.test_set),
            batch_size=batch_size,
            shuffle=True,
            device=device,
            sort_key=lambda x: data.interleave_keys(len(x.user), len(x.movie)),
            sort_within_batch=True,
            repeat=True)

        # self.train_iter = data.Iterator.splits(
        #     self.train_set,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     device=device,
        #     sort_key=lambda x: data.interleave_keys(len(x.user), len(x.movie)),
        #     sort_within_batch=True,
        #     repeat=True)
        # self.validation_iter = data.Iterator.splits(
        #     self.validation_set,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     device=device,
        #     sort_key=lambda x: data.interleave_keys(len(x.user), len(x.movie)),
        #     sort_within_batch=True,
        #     repeat=False)
        # self.test_iter = data.Iterator.splits(
        #     self.test_set,
        #     batch_size=batch_size,
        #     shuffle=True,
        #     device=device,
        #     sort_key=lambda x: data.interleave_keys(len(x.user), len(x.movie)),
        #     sort_within_batch=True,
        #     repeat=False)

        # self.user.build_vocab(self.train_set)
        # self.movie.build_vocab(self.train_set)
        # self.rating.build_vocab(self.train_set)



class MovieLens:
    """
    Class to handle the MovieLens data
    """

    def __init__(self, device, path='./Datasets/MovieLens-Small/ratings.csv'):
        print('Device: ' + str(device))

        self.path = path

        self.user = data.Field(sequential=False, use_vocab=True, unk_token=None)
        self.movie = data.Field(sequential=False, use_vocab=True)
        self.rating = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

        self.train_set, self.validation_set, self.test_set = data.TabularDataset(
            path=path,
            format='csv',
            fields=[('user', self.user), ('movie', self.movie), ('rating', self.rating), ('timestamp', None)],
            skip_header=True,
        ).split(split_ratio=[0.9, 0.05, 0.05])

        self.train_iter, self.validation_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_set, self.validation_set, self.test_set),
            batch_size=128,
            shuffle=True,
            device=device,
            sort_key=lambda x: len(x.movie),
            repeat=True
        )

        self.user.build_vocab(self.train_set)
        self.movie.build_vocab(self.train_set)

    def get_train_set(self):
        return self.train_set

    def get_validation_set(self):
        return self.validation_set

    def get_test_set(self):
        return self.test_set

    def get_train_iter(self):
        return self.train_iter

    def get_validation_iter(self):
        return self.validation_iter

    def get_test_iter(self):
        return self.test_iter

    def get_ratings_matrix(self):
        ratings_df = pd.read_csv(filepath_or_buffer=self.path)
        R_df = ratings_df.pivot(index='userId', columns='movieId', values='rating').fillna(0)
        return R_df, R_df.columns

    def get_normalized_ratings_matrix(self):
        R, _ = self.get_ratings_matrix()
        R = R.as_matrix()
        user_ratings_mean = np.mean(R, axis=1)
        R_demeaned = R - user_ratings_mean.reshape(-1, 1)
        return R_demeaned, user_ratings_mean


class TalentFox:
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


class citeulike:
    """
    Class to handle the Cite-U-Like data

    Predict:
    match_status

    """

    def __init__(self, batch_size=100):
        print('Device: ' + str(device))

        self.user = data.Field(sequential=False, use_vocab=False)
        self.doc_title = data.Field(sequential=True, lower=True, include_lengths=True)
        self.ratings = data.Field(sequential=False, use_vocab=False)
        # self.doc_abstract = data.Field(sequential=True, tokenize=tokenizer, lower=True)

        self.train_set, self.validation_set, self.test_set = data.TabularDataset.splits(
            path='./Datasets/citeulike/',
            train='train_data.csv',
            validation='val_data.csv',
            test='test_data.csv',
            format='csv',
            fields=[
                ('index', None),
                ('user', self.user),
                ('doc_id', None),
                ('ratings', self.ratings),
                ('doc_title', self.doc_title),
                # ('doc_abstract', self.doc_abstract)
            ],
            skip_header=True,
        )

        self.train_iter, self.validation_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_set, self.validation_set, self.test_set),
            batch_size=batch_size,
            shuffle=True,
            device=device,
            sort_key=lambda x: len(x.doc_title),
            sort_within_batch=True,
            repeat=True)

        self.user.build_vocab(self.train_set)
        self.ratings.build_vocab(self.train_set)
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        # self.doc_abstract.build_vocab(self.train_set, max_size=None, vectors=vocab.Vectors('wiki.simple.vec', url=url))
        self.doc_title.build_vocab(self.train_set, max_size=None, vectors=vocab.Vectors('wiki.simple.vec', url=url))


def to_csv_citeulike(total=0):
    docs = pd.read_csv('Datasets/citeulike/raw-data.csv', usecols=['doc.id', 'raw.title', 'raw.abstract'],
                       dtype={'doc.id': np.int32, 'raw.title': str, 'raw.abstract': str}, header=0, sep=',')
    users = pd.read_csv('Datasets/citeulike/user-info.csv', usecols=['user.id', 'doc.id', 'rating'], header=0,
                        sep=',')
    docs.set_index('doc.id', inplace=True)
    titles, abstracts, users_list, ratings, docs_list = [], [], [], [], []
    test_titles, test_abstracts, test_users_list, test_ratings, test_docs_list = [], [], [], [], []
    val_titles, val_abstracts, val_users_list, val_ratings, val_docs_list = [], [], [], [], []
    cnt = 0
    max_user = users.iloc[-1]['user.id'] if total is 0 else users.iloc[total]['user.id']
    for index, row in users.iterrows():
        cnt += 1
        if total is not 0:
            if cnt == total:
                break
        if cnt % 9 == 0:
            test_titles.append(docs.loc[row['doc.id']]['raw.title'])
            test_abstracts.append(docs.loc[row['doc.id']]['raw.abstract'])
            test_users_list.append(row['user.id'] - 1)
            test_ratings.append(1)
            test_docs_list.append(row['doc.id'])
            continue
        if cnt % 8 == 0:
            val_titles.append(docs.loc[row['doc.id']]['raw.title'])
            val_abstracts.append(docs.loc[row['doc.id']]['raw.abstract'])
            val_users_list.append(row['user.id'] - 1)
            val_ratings.append(1)
            val_docs_list.append(row['doc.id'])
            continue

        titles.append(docs.loc[row['doc.id']]['raw.title'])
        abstracts.append(docs.loc[row['doc.id']]['raw.abstract'])
        users_list.append(row['user.id'] - 1)
        ratings.append(row['rating'])
        docs_list.append(row['doc.id'])
        titles.append(docs.loc[row['doc.id']]['raw.title'])
        abstracts.append(docs.loc[row['doc.id']]['raw.abstract'])
        users_list.append(randint(0, max_user))
        ratings.append(0)
        docs_list.append(row['doc.id'])

    d = {'user.id': users_list, 'doc.id': docs_list, 'rating': ratings, 'raw.title': titles,
         'raw.abstract': abstracts}
    df = pd.DataFrame(data=d)
    df.to_csv('Datasets/citeulike/train_data.csv')
    d = {'user.id': val_users_list, 'doc.id': val_docs_list, 'rating': val_ratings, 'raw.title': val_titles,
         'raw.abstract': val_abstracts}
    df = pd.DataFrame(data=d)
    df.to_csv('Datasets/citeulike/val_data.csv')
    d = {'user.id': test_users_list, 'doc.id': test_docs_list, 'rating': test_ratings, 'raw.title': test_titles,
         'raw.abstract': test_abstracts}
    df = pd.DataFrame(data=d)
    df.to_csv('Datasets/citeulike/test_data.csv')


def to_csv_movielens(total=0):
    users = pd.read_csv('Datasets/MovieLens-Small/ratings.csv', usecols=['userId', 'movieId', 'rating'],
                       dtype={'userId': np.int32, 'movieId': np.int32, 'rating': np.float64}, header=0, sep=',')

    users.set_index(['userId', 'movieId'])

    users_list, movies_list, ratings_list = [], [], []
    test_users_list, test_movies_list, test_ratings_list = [], [], []
    val_users_list, val_movies_list, val_ratings_list = [], [], []

    cnt = 0
    max_user = users.iloc[-1]['userId'] if total is 0 else users.iloc[total]['userId']

    for index, row in users.iterrows():
        cnt += 1
        if total is not 0:
            if cnt == total:
                break
        if cnt % 9 == 0:
            test_ratings_list.append(row['rating'])
            test_movies_list.append(row['movieId'])
            test_users_list.append(row['userId'])
            continue
        if cnt % 8 == 0:
            val_ratings_list.append(row['rating'])
            val_movies_list.append(row['movieId'])
            val_users_list.append(row['userId'])
            continue


        ratings_list.append(row['rating'])
        movies_list.append(row['movieId'])
        users_list.append(row['userId'])


    d = {'user': users_list, 'movie': movies_list, 'rating': ratings_list}
    df = pd.DataFrame(data=d)
    df.to_csv('Datasets/MovieLens-Small/train_data.csv')
    d = {'user': val_users_list, 'movie': val_movies_list, 'rating': val_ratings_list}
    df = pd.DataFrame(data=d)
    df.to_csv('Datasets/MovieLens-Small/val_data.csv')
    d = {'user': test_users_list, 'movie': test_movies_list, 'rating': test_ratings_list}
    df = pd.DataFrame(data=d)
    df.to_csv('Datasets/MovieLens-Small/test_data.csv')


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #dataset = MovieLens(device=device)
    #user = dataset.user
    #print(user.vocab.itos)
    #to_csv_movielens()
    dataset = MovieLens(device=device)
    ratings, columns = dataset.get_ratings_matrix()
    R_demeaned, user_ratings_mean = dataset.get_normalized_ratings_matrix()
    #print(ratings)

    import seaborn as sns
    import matplotlib.pyplot as plt
    ratings_to_plot = R_demeaned + user_ratings_mean.reshape(-1, 1)
    #cmap = sns.cm.gist_heat_r
    ax = sns.heatmap(data=ratings_to_plot[0:50, 0:100], vmin=0.0, vmax=5.0, cmap='hot_r')
    ax.set(xlabel='MovieId', ylabel='UserId')
    plt.show()