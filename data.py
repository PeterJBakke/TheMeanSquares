"""
Data pre-processing file
"""

import pandas as pd
import numpy as np
import os
from torchtext import data, vocab
import torch
import spacy

spacy_en = spacy.load('en')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MovieLens:
    """
    Class to handle the MovieLens data
    """

    def __init__(self, device, path='./Datasets/MovieLens-Small/ratings.csv'):
        print('Device: ' + str(device))

        self.user = data.Field(sequential=False, use_vocab=True)
        self.movie = data.Field(sequential=False, use_vocab=True)
        self.rating = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

        self.train_set, self.validation_set, self.test_set = data.TabularDataset(
            path=path,
            format='csv',
            fields=[('user', self.user), ('movie', self.movie), ('rating', self.rating), ('timestamp', None)],
            skip_header=True,
        ).split(split_ratio=[0.7, 0.15, 0.15])

        self.train_iter, self.validation_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_set, self.validation_set, self.test_set),
            batch_size=100,
            device=device,
            sort_key=lambda x: len(x.movie))

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

    def __init__(self):
        print('Device: ' + str(device))

        self.user = data.Field(sequential=False, use_vocab=True)
        self.doc = data.Field(sequential=False, use_vocab=False)
        self.rating = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

        self.train_set, self.validation_set, self.test_set = data.TabularDataset(
            path='./Datasets/citeulike/user-info.csv',
            format='csv',
            fields=[('id', None), ('user', self.user), ('doc', self.doc), ('rating', self.rating), ('timestamp', None)],
            skip_header=True,
        ).split(split_ratio=[0.9, 0.05, 0.05])

        self.docs = pd.read_csv('Datasets/citeulike/raw-data.csv',
                                usecols=['doc.id', 'raw.title', 'raw.abstract'],
                                dtype={'doc.id': np.int32, 'raw.title': str, 'raw.abstract': str}, header=0,
                                sep=',')

        self.docs.set_index('doc.id', inplace=True)

        self.train_iter, self.validation_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_set, self.validation_set, self.test_set),
            batch_size=100,
            # shuffle=True,
            device=device,
            sort_key=lambda x: int(x.user))

        self.user.build_vocab(self.train_set)
        self.doc.build_vocab(self.train_set)

    def get_document_abstract(self, docID):
        return self.docs.loc[docID]['raw.abstract']


def load_vocab():
    text = data.Field(sequential=True, tokenize=tokenizer, lower=True)

    dataset = data.TabularDataset(
        path='./Datasets/citeulike/raw-data.csv',
        format='csv',
        fields=[
            ('doc', None),
            ('title', None),
            ('citeulike-id', None),
            ('raw-title', None),
            ('text', text)
        ],
        skip_header=True)

    url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
    text.build_vocab(dataset, max_size=None, vectors=vocab.Vectors('wiki.simple.vec', url=url))

    return text.vocab


class citeulike_merged:
    """
    Class to handle the Cite-U-Like data

    Predict:
    match_status

    """

    def __init__(self):
        print('Device: ' + str(device))

        self.user = data.Field(sequential=False, use_vocab=True)
        self.doc_title = data.Field(sequential=False, use_vocab=False, dtype=torch.float)
        self.doc_abstract = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

        self.train_set, self.validation_set, self.test_set = data.TabularDataset(
            path='./Datasets/citeulike/user-info.csv',
            format='csv',
            fields=[('id', None), ('user', self.user), ('doc_title', self.doc_title),
                    ('doc_abstract', self.doc_abstract)],
            skip_header=True,
        ).split(split_ratio=[0.8, 0.1, 0.1])

        self.train_iter, self.validation_iter, self.test_iter = data.BucketIterator.splits(
            (self.train_set, self.validation_set, self.test_set),
            batch_size=100,
            # shuffle=True,
            device=device,
            sort_key=lambda x: int(x.user))

        self.user.build_vocab(self.train_set)

        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.simple.vec'
        self.doc_abstract.build_vocab(self.train_set, max_size=None, vectors=vocab.Vectors('wiki.simple.vec', url=url))
        self.doc_title.build_vocab(self.train_set, max_size=None, vectors=vocab.Vectors('wiki.simple.vec', url=url))


def tokenizer(text):  # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]


if __name__ == "__main__":
    dataset = citeulike()
    train_iter = dataset.train_iter
    print(train_iter.device)
    print(train_iter.epoch)

    vocab = load_vocab()
