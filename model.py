"""
Model file
"""
import torch
import torch.nn as nn
import numpy as np

max_rating = 5.0
min_rating = 0.5


class EmbeddingNet(nn.Module):
    def __init__(self, user_field, movie_field, n_factors=10, hidden=10, p1=0.5, p2=0.5):
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
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
        np_movies = np.asarray([self.movie_field.vocab.stoi[str(movie)] for movie in movies.cpu().data.numpy()],
                                   dtype=int)

        movie_numbers = torch.from_numpy(np_movies).to(self.device).long()
        return self.m(movie_numbers)

    def get_user_embedding(self, users):
        if torch.cuda.is_available():
            np_users = np.asarray([self.user_field.vocab.stoi[str(user)] for user in users.cpu().data.numpy()])
        else:
            np_users = np.asarray([self.user_field.vocab.stoi[str(user)] - 1 for user in users.cpu().data.numpy()])
        user_numbers = torch.from_numpy(np_users).to(self.device).long()
        return self.u(user_numbers)

    def forward(self, batch):
        x = torch.cat([self.get_user_embedding(batch.user), self.get_movie_embedding(batch.movie)], dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return torch.sigmoid(x) * (max_rating - min_rating + 1) + min_rating - 0.5
