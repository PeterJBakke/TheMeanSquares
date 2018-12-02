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


def train_with_negative_sampling(train_iter, val_iter, net, test_iter, optimizer, criterion, num_user, num_epochs=5):
    net.train()
    prev_epoch = 0
    train_loss = []
    train_error = []
    train_accs = []

    val_res = []

    for batch in train_iter:

        net.train()

        pos_and_neg_batch = negative_sampling(batch, num_user)
        target_pos = Variable(torch.tensor([1 for _ in range(len(batch))]).cuda())
        target_neg = Variable(torch.tensor([0 for _ in range(len(batch))]).cuda())

        output = net(pos_and_neg_batch).reshape(-1)
        target = torch.cat((target_pos, target_neg), 0).float().cuda()
        batch_loss = criterion(output, target)

        train_loss.append(get_numpy(batch_loss))
        train_error.append(accuracy_sigmoid(output, target))
        train_accs.append(accuracy(output, target))

        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        # print(
        #     "Train loss: {:.3f}, Train avg error: {:.3f}"
        #         .format(criterion(output, target), accuracy_sigmoid(output, target)))

        if train_iter.epoch != prev_epoch:
            net.eval()
            val_loss, val_accs, val_length = [0, 0, 0]

            for val_batch in val_iter:
                val_output = net(val_batch).reshape(-1)
                val_target = Variable(torch.tensor([1 for _ in range(len(val_batch))]).float().cuda())
                val_loss += criterion(val_output, val_target) * val_batch.batch_size
                val_accs += accuracy_sigmoid(val_output, val_target) * val_batch.batch_size
                val_length += val_batch.batch_size

            val_loss /= val_length
            val_accs /= val_length
            val_res.append(val_accs)

            print(
                "Epoch {}: Train loss: {:.2f},  Train accs: {:.2f}, Train avg error: {:.2f}"
                    .format(train_iter.epoch, np.mean(train_loss), 1.0 - np.mean(train_accs), np.mean(train_error)))
            print(
                "          Validation loss: {:.2f}, Validation avg error: {:.2f}"
                    .format(val_loss, val_accs))
            print()
            train_loss = []
            train_error = []
            train_accs = []
            # plot_res(train_res, val_res, train_iter.epoch)

            net.train()

        prev_epoch = train_iter.epoch
        if train_iter.epoch == num_epochs:
            break


def negative_sampling(batch, num_user):
    random_user = torch.tensor(
        [randint(0, num_user) for _ in range(len(batch))]
    ).cuda()

    author = torch.cat((batch.user, random_user), 0).cuda()
    doc_title = torch.cat((batch.doc_title, batch.doc_title), 1).cuda()

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


def accuracy_sigmoid(output, target):
    return torch.mean(torch.abs(output - target).float()).cpu().data.numpy()


def accuracy(output, target):
    return torch.mean(torch.abs(torch.round(output) - target)).cpu().data.numpy()


def print_params(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def get_numpy(loss):
    return loss.cpu().data.numpy()
