"""
Main
"""

import torch
from torch import optim, nn
from model import MovieLensNet, CiteULikeModel, LstmNet, TalentNet, CuLMFNet
from data import MovieLens, citeulike, TalentFox
from train import movie_lens_train, train_with_negative_sampling, talent_fox_train

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

movie_lens_train(train_iter=train_set, test_iter=test_set, val_iter=validation_set,
                 net=net, optimizer=opt, criterion=criterion, num_epochs=50)

############################################################

# citeulike = citeulike()
# text_vocab = load_vocab()
# text_vectors = text_vocab.vectors
# num_users = len(citeulike.user.vocab.itos)
#
# train_iter = citeulike.train_iter
# test_iter = citeulike.test_iter
# validation_iter = citeulike.validation_iter
#
# user_field = citeulike.user
# doc_field = citeulike.doc
#
# net = CiteULikeModel(text_vectors=text_vectors, user_field=user_field, user_dim=10).to(device)
# opt = optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-5)
# criterion = nn.BCELoss()
# train_with_negative_sampling(citeulike, train_iter=train_iter, test_iter=test_iter, val_iter=validation_iter,
#                              net=net, optimizer=opt, criterion=criterion, num_epochs=50, num_user=num_users,
#                              text_stoi=text_vocab.stoi)

##############################################################

"""
citeulike = citeulike(batch_size=200)

train_iter = citeulike.train_iter
test_iter = citeulike.test_iter
validation_iter = citeulike.validation_iter

largest_doc_id = max(list(map(int, train_iter.dataset.fields['doc_id'].vocab.itos[1:])))

user_field = citeulike.user
title_field = citeulike.doc_id

net = CuLMFNet(article_field=title_field, user_field=user_field, num_docs=largest_doc_id).to(device)
opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCELoss()
train_with_negative_sampling(train_iter=train_iter, test_iter=test_iter, val_iter=validation_iter,
                                net=net, optimizer=opt, criterion=criterion, num_epochs=50)

tf = TalentFox()

train_iter = tf.train_iter
test_iter = tf.test_iter
val_iter = tf.validation_iter

job_title = tf.job_title
candidate_title = tf.candidate_title

net = TalentNet(job_title=job_title, candidate_title=candidate_title).to(device)
opt = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.BCELoss()

talent_fox_train(train_iter=train_iter, test_iter=test_iter, val_iter=val_iter, net=net, optimizer=opt, criterion=criterion, num_epochs=5)
"""