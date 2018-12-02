"""
Main
"""

import torch
from torch import optim, nn
from model import MovieLensNet, CiteULikeModel
from data import MovieLens, citeulike, load_vocab
from train import movie_lens_train, cite_u_like_train

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# movie_data = MovieLens(device=device)
#
# train_set = movie_data.get_train_iter()
# test_set = movie_data.get_test_iter()
# validation_set = movie_data.get_validation_iter()
#
# user_field = movie_data.user
# movie_field = movie_data.movie
#
# net = MovieLensNet(user_field=user_field, movie_field=movie_field, device=device, n_factors=10).to(device)
#
# opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
#
# criterion = nn.MSELoss()
#
# movie_lens_train(train_iter=train_set, test_iter=test_set, val_iter=validation_set,
#       net=net, optimizer=opt, criterion=criterion, num_epochs=50)


citeulike = citeulike()
text_vocab = load_vocab()
text_vectors = text_vocab.vectors
num_users = len(citeulike.user.vocab.itos)

train_iter = citeulike.train_iter
test_iter = citeulike.test_iter
validation_iter = citeulike.validation_iter

user_field = citeulike.user
doc_field = citeulike.doc

net = CiteULikeModel(text_vectors=text_vectors, user_field=user_field, user_dim=10).to(device)
opt = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.BCELoss()
cite_u_like_train(citeulike, train_iter=train_iter, test_iter=test_iter, val_iter=validation_iter,
                  net=net, optimizer=opt, criterion=criterion, num_epochs=50, num_user=num_users,
                  text_stoi=text_vocab.stoi)
