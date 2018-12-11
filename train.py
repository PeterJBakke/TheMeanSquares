from types import SimpleNamespace
import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from random import randint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def movie_lens_train(train_iter, val_iter, net, test_iter, optimizer, criterion, num_epochs=5):
    net.train()
    prev_epoch = 0
    for batch in train_iter:
        if train_iter.epoch != prev_epoch:
            net.eval()
            val_loss, val_accs, val_length = [0, 0, 0]

            for val_batch in val_iter:
                val_output = net(val_batch)
                val_loss += criterion(val_output.reshape(-1), val_batch.rating) * val_batch.batch_size
                val_accs += accuracy(val_output, val_batch.rating) * val_batch.batch_size
                val_length += val_batch.batch_size

            val_loss /= val_length
            val_accs /= val_length

            print("Epoch {}:  Loss: {:.2f}, Avg distance from target: {:.2f}".format(train_iter.epoch, val_loss,
                                                                                     val_accs))
            net.train()

        net.train()
        output = net(batch)
        batch_loss = criterion(output.reshape(-1), batch.rating)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        prev_epoch = train_iter.epoch
        if train_iter.epoch == num_epochs:
            break

def talent_fox_train(train_iter, val_iter, net, test_iter, optimizer, criterion, num_epochs=5):
    net.train()
    prev_epoch = 0
    train_loss = []
    train_accs = []
    train_accs_pos = 0
    train_sum = 0
    val_res = []
    for batch in train_iter:
        job_title, candidate_title, match_status = batch.job_title, batch.candidate_title, batch.match_status
        net.train()

        batch_sampling = {'job_title': job_title, 'candidate_title': candidate_title}
        output = net(SimpleNamespace(**batch_sampling)).reshape(-1)
        targets = match_status.float().to(device)
        criterion.weight = weights(targets)
        batch_loss = criterion(output, targets)

        train_loss.append(get_numpy(batch_loss))
        train_accs.append(accuracy_sigmoid(output, targets))
        train_accs_pos += accuracy_talent(output, targets)
        train_sum += sum_targets(targets)

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()


        if train_iter.epoch != prev_epoch:
            net.eval()
            val_loss, val_accs, val_accs_pos, val_length, val_sum = 0, 0, 0, 0, 0

            for val_batch in val_iter:
                if val_iter.epoch != train_iter.epoch-1:
                    break
                job_title, candidate_title, match_status = batch.job_title, val_batch.candidate_title, val_batch.match_status
                batch_sampling = {'job_title': job_title, 'candidate_title': candidate_title}
                val_output = net(SimpleNamespace(**batch_sampling)).reshape(-1)
                val_target = match_status.float().to(device)
                val_loss += criterion(val_output, val_target) * val_batch.batch_size
                val_accs += accuracy_sigmoid(val_output, val_target) * val_batch.batch_size
                val_accs_pos += accuracy_talent(val_output, val_target)
                val_length += val_batch.batch_size
                val_sum += sum_targets(val_target)

            val_loss /= val_length
            val_accs /= val_length
            val_res.append(val_accs)

            print(
                "Epoch {}: Train loss: {:.2f},  Train accs total: {:.2f}, Train accs positive: {}/{}"
                    .format(train_iter.epoch, np.mean(train_loss), 1.0 - np.mean(train_accs), train_accs_pos, train_sum))
            print(
                "          Validation loss: {:.2f}, Validation accs total: {:.2f}, Validation accs positive: {}/{}"
                    .format(val_loss, 1.0 - val_accs, val_accs_pos, val_sum))
            print()
            train_loss = []
            train_accs = []
            train_accs_pos = 0
            train_sum = 0
            net.train()

        prev_epoch = train_iter.epoch
        if train_iter.epoch == num_epochs:
            break


def train_with_negative_sampling(train_iter, val_iter, net, test_iter, optimizer, criterion, num_epochs=5):
    net.train()
    prev_epoch = 0
    train_loss = []
    train_error = []
    train_accs = []
    val_res = []
    for batch in train_iter:
        users,(docs, lengths), ratings = batch.user, batch.doc_title, batch.ratings
        net.train()

        batch_with_negative_sampling = {'user': users, 'doc_title': docs}
        output = net(SimpleNamespace(**batch_with_negative_sampling), lengths).reshape(-1)
        targets = ratings.float().to(device)
        batch_loss = criterion(output, targets)

        train_loss.append(get_numpy(batch_loss))
        train_error.append(accuracy_sigmoid(output, targets))
        train_accs.append(accuracy(output, targets))

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # print(
        #     "Train loss: {:.3f}, Train avg error: {:.3f}"
        #         .format(criterion(output, target), accuracy_sigmoid(output, target)))

        if train_iter.epoch != prev_epoch:
            net.eval()
            val_loss, val_accs, val_err, val_length = [0, 0, 0, 0]

            for val_batch in val_iter:
                if val_iter.epoch != train_iter.epoch-1:
                    break
                users, (docs, lengths), ratings = val_batch.user, val_batch.doc_title, val_batch.ratings
                batch_without_negative_sampling = {'user': users, 'doc_title': docs}
                val_output = net(SimpleNamespace(**batch_without_negative_sampling), lengths).reshape(-1)
                val_target = ratings.float().to(device)
                val_loss += criterion(val_output, val_target) * val_batch.batch_size
                val_err += accuracy_sigmoid(val_output, val_target) * val_batch.batch_size
                val_accs += accuracy_sigmoid(val_output, val_target) * val_batch.batch_size
                val_length += val_batch.batch_size

            val_loss /= val_length
            val_err /= val_length
            val_accs /= val_length
            val_res.append(val_accs)

            print(
                "Epoch {}: Train loss: {:.2f},  Train accs: {:.2f}, Train avg error: {:.2f}"
                    .format(train_iter.epoch, np.mean(train_loss), 1.0 - np.mean(train_accs), np.mean(train_error)))
            print(
                "          Validation loss: {:.2f}, Validation accs: {:.2f}, Validation avg error: {:.2f}"
                    .format(val_loss, 1.0 - val_accs, val_err))
            print()
            train_loss = []
            train_error = []
            train_accs = []
            # plot_res(train_res, val_res, train_iter.epoch)

            net.train()

        prev_epoch = train_iter.epoch
        if train_iter.epoch == num_epochs:
            break

def negative_sampling(users, docs, num_user):
    if torch.cuda.is_available():
        random_user = torch.tensor(
        [randint(0, num_user) for _ in range(len(users))]
        ).to(device)
    else:
        random_user = torch.tensor(
            [randint(0, num_user-1) for _ in range(len(users))]
        ).to(device)

    author = torch.cat((users, random_user), 0).to(device)
    doc_title = torch.cat((docs, docs), 1).to(device)

    batch_with_negative_sampling = {'user': author, 'doc_title': doc_title}
    return SimpleNamespace(**batch_with_negative_sampling)

def plot_res(train_res, val_res, num_res):
    x_vals = np.arange(num_res)
    plt.figure()
    plt.plot(x_vals, train_res, 'r', x_vals, val_res, 'b')
    plt.legend(['Train Accucary', 'Validation Accuracy'])
    plt.xlabel('Updates'), plt.ylabel('Acc')

def accuracy_one_hot(output, target):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = torch.eq(torch.max(output, 1)[1], target)
    # averaging the one-hot encoded vector
    return torch.mean(correct_prediction.float())

def sum_targets(targets):
    return torch.sum(targets)

def accuracy_talent(output, targets):
    correct_predictions = 0
    for idx, val in enumerate(output):
        if val > 0.5 and targets[idx] == torch.tensor(1.0):
            correct_predictions += 1
    return correct_predictions

def accuracy_sigmoid(output, target):
    return torch.mean(torch.abs(output - target).float()).cpu().data.numpy()


def accuracy(output, target):
    return torch.mean(torch.abs(torch.round(output) - target)).cpu().data.numpy()

def weights(target):
    weight = []
    for val in target:
        if val == torch.tensor(1.):
            weight.append(100.)
        else:
            weight.append(1.)
    return torch.tensor(weight)

def print_params(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def get_numpy(loss):
    return loss.cpu().data.numpy()
