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


class MovieLens:
    """
    Class to handle the MovieLens data
    """

    def __init__(self, device, path='./Datasets/MovieLens-Small/ratings.csv', batch_size=100):
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
            batch_size=batch_size,
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

    def __init__(self, batch_size=100):
        print('Device: ' + str(device))

        self.candidate_title = data.Field(sequential=True, lower=True, tokenize=tokenizer, include_lengths=True, use_vocab=True)
        self.candidate_resume = data.Field(sequential=True, lower=True, tokenize=tokenizer, include_lengths=True, use_vocab=True)
        self.job_title = data.Field(sequential=True, lower=True, tokenize=tokenizer, include_lengths=True, use_vocab=True)
        self.job_description = data.Field(sequential=True, lower=True, tokenize=tokenizer, include_lengths=True, use_vocab=True)
        self.match_status = data.Field(sequential=False, use_vocab=False)

        self.train_set, self.validation_set = data.TabularDataset.splits(
            path='./Datasets/talentfox_match_data/',
            train='train_data.csv',
            validation='val_data.csv',
            format='csv',
            fields=[
                ('index', None),
                ('job_title', self.job_title),
                ('job_description', self.job_description),
                ('candidate_title', self.candidate_title),
                ('candidate_resume', self.candidate_resume),
                ('match_status', self.match_status)
            ],
            skip_header=True,
        )

        self.train_iter, self.validation_iter = data.BucketIterator.splits(
            (self.train_set, self.validation_set),
            batch_size=batch_size,
            shuffle=True,
            device=device,
            sort_key=lambda x: len(x.job_title),
            sort_within_batch=True,
            repeat=True)

        self.match_status.build_vocab(self.train_set)
        url = 'https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.de.vec'
        self.job_title.build_vocab(self.train_set, vectors=vocab.Vectors('wiki.de.vec', url=url))
        self.job_description.build_vocab(self.train_set, vectors=vocab.Vectors('wiki.de.vec', url=url))
        self.candidate_title.build_vocab(self.train_set, vectors=vocab.Vectors('wiki.de.vec', url=url))
        self.candidate_resume.build_vocab(self.train_set, vectors=vocab.Vectors('wiki.de.vec', url=url))

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

        self.train_set, self.validation_set = data.TabularDataset.splits(
            path='./Datasets/citeulike/',
            train='train_data.csv',
            validation='val_data.csv',
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

        self.train_iter, self.validation_iter = data.BucketIterator.splits(
            (self.train_set, self.validation_set),
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
    val_titles, val_abstracts, val_users_list, val_ratings, val_docs_list = [], [], [], [], []
    cnt = 0
    max_user = users.iloc[-1]['user.id'] if total is 0 else users.iloc[total]['user.id']
    for index, row in users.iterrows():
        cnt += 1
        if total is not 0:
            if cnt == total:
                break
        if cnt % 5 == 0:
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

def to_csv_talentfox(total=0):
    keys = ('mba', 'it', 'ceo', 'ad', 'bdm', 'kam', 'ka', 'dr.', 'bwl', 'oa', 'csm')
    docs = pd.read_csv('Datasets/talentfox_match_data/processed_dataset.csv', header=0, sep=',')
    job_description, job_title, candidate_resume, candidate_title, match_status = [], [], [], [], []
    val_job_description, val_job_title, val_candidate_resume, val_candidate_title, val_match_status = [], [], [], [], []
    cnt = 0
    for index, row in docs.iterrows():
        if len(str(docs.loc[index]['candidate_resume'])) < 4:
            continue
        if len(str(docs.loc[index]['job_description'])) < 4:
            continue
        if len(str(docs.loc[index]['job_title'])) < 4:
            if str(docs.loc[index]['job_title']).lower() not in keys:
                continue
        if len(str(docs.loc[index]['candidate_title'])) < 4:
            if str(docs.loc[index]['candidate_title']).lower() not in keys:
                continue
        try:
            match = 1 if int(docs.loc[index]['match_status']) >= 4 else 0
        except:
            continue
        cnt += 1
        if total is not 0:
            if cnt == total:
                break
        if cnt % 5 == 0:
            val_job_description.append(docs.loc[index]['job_description'])
            val_job_title.append(docs.loc[index]['job_title'])
            val_candidate_resume.append(docs.loc[index]['candidate_resume'])
            val_candidate_title.append(docs.loc[index]['candidate_title'])
            val_match_status.append(match)
            continue
        job_description.append(docs.loc[index]['job_description'])
        job_title.append(docs.loc[index]['job_title'])
        candidate_resume.append(docs.loc[index]['candidate_resume'])
        candidate_title.append(docs.loc[index]['candidate_title'])
        match_status.append(match)

    d = {'job_title': job_title, 'job_description': job_description, 'candidate_title': candidate_title, 'candidate_resume': candidate_resume, 'match_status': match_status}
    df = pd.DataFrame(data=d)
    df.to_csv('Datasets/talentfox_match_data/train_data.csv')

    d = {'job_title': val_job_title, 'job_description': val_job_description, 'candidate_title': val_candidate_title, 'candidate_resume': val_candidate_resume, 'match_status': val_match_status}
    df = pd.DataFrame(data=d)
    df.to_csv('Datasets/talentfox_match_data/val_data.csv')

STOP_WORDS = {'(', ')', '/', 'm', 'w', '-', ' ', '.', '\t'}

def tokenizer(text):  # create a tokenizer function
    tokens = [tok.text for tok in spacy_en.tokenizer(text)]
    tokens = list(filter(lambda token: token not in STOP_WORDS, tokens))
    return tokens

if __name__ == "__main__":
    dataset = citeulike()
    train_iter = dataset.train_iter
    print(train_iter.device)
    print(train_iter.epoch)

    vocab = load_vocab()
