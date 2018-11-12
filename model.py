"""
Model file
"""
import torch
import torch.nn as nn
from torch.nn import functional as f

max_rating = 5.0
min_rating = 0.5


class EmbeddingNet(nn.Module):
    def __init__(self, n_users, n_movies, n_factors=10, hidden=10, p1=0.5, p2=0.5):
        super().__init__()
        # (self.u, self.m) = [get_emb(*o) for o in [
        #     (n_users, n_factors), (n_movies, n_factors)]]
        self.u = nn.Embedding(n_users, n_factors)
        self.u.weight.data.uniform_(0, 0.05)

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

    def forward(self, cats):
        users, movies = cats[:, 0], cats[:, 1]
        print(users)

        x = torch.cat([self.u(users), self.m(movies)], dim=1)
        x = self.lin1(x)
        x = self.lin2(x)
        return f.sigmoid(x) * (max_rating - min_rating + 1) + min_rating - 0.5
