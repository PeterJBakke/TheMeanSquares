"""
Model file
"""
import torch
import torch.nn as nn
from torch.nn import functional as f
import numpy as np

max_rating = 5.0
min_rating = 0.5


class EmbeddingNet(nn.Module):
    def __init__(self, user_field, movie_field, n_factors=10, hidden=10, p1=0.5, p2=0.5):
        super().__init__()

        self.movie_field = movie_field
        self.user_field = user_field

        n_users = len(self.user_field.vocab.freqs)
        self.u = nn.Embedding(n_users, n_factors)
        self.u.weight.data.uniform_(0, 0.05)

        n_movies = len(self.movie_field.vocab.freqs)
        self.m = nn.Embedding(n_movies, n_factors)
        self.m.weight.data.uniform_(0, 0.05)

        self.lin1 = nn.Sequential(
            nn.Dropout(p1),
            nn.Linear(n_factors * 2, hidden),
            nn.ReLU(),
        )

        self.lin2 = nn.Sequential(
            nn.Dropout(p2),
            nn.Linear(hidden, 1),
        )
        print(self)

    def get_movie_embedding(self, movies):
        np_movies = np.asarray([self.movie_field.vocab.stoi[movie] for movie in movies])
        movie_number = torch.from_numpy(np_movies)
        return self.m(movie_number)

    def get_user_embedding(self, users):
        np_users = np.asarray([self.user_field.vocab.stoi[user] for user in users])
        user_number = torch.from_numpy(np_users)
        return self.u(user_number)

    def forward(self, batch):
        x = torch.cat([self.get_user_embedding(batch.user), self.get_movie_embedding(batch.movie)], dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return f.sigmoid(x) * (max_rating - min_rating + 1) + min_rating - 0.5


