"""
Main
"""

import torch
from torch import optim, nn
from model import MovieLensNet
from data import MovieLens
from train import train

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

movie_data = MovieLens(device=device)

train_set = movie_data.get_train_iter()
test_set = movie_data.get_test_iter()
validation_set = movie_data.get_validation_iter()

user_field = movie_data.user
movie_field = movie_data.movie

net = MovieLensNet(user_field=user_field, movie_field=movie_field, device=device, n_factors=10).to(device)

opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)

criterion = nn.MSELoss()

train(train_iter=train_set, test_iter=test_set, val_iter=validation_set,
      net=net, optimizer=opt, criterion=criterion, num_epochs=50)
