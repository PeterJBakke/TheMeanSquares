"""
Main
"""

from torch import optim, nn
from model import EmbeddingNet
from data import MovieLens
from train import train

myData = MovieLens()
movies = myData.MovieLensMoviesData()
ratings = myData.MovieLensRatingsData()
users = myData.MovieLensUsers()

train_set = ratings.values[:75000, :3]
test_set = ratings.values[75000:, :3]

n_users = int(ratings.userId.nunique())
n_movies = int(ratings.movieId.nunique())

net = EmbeddingNet(n_users, n_movies, n_factors=10).cuda()
opt = optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()

train(train_set=train_set, test_set=test_set, net=net, optimizer=opt, criterion=criterion, num_epochs=5)
