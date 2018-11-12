import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

# we could have done this ourselves,
# but we should be aware of sklearn and it's tools
from sklearn.metrics import accuracy_score


def train(train_set, test_set, net, optimizer, criterion, batch_size=100, num_epochs=5):
    num_samples_train = train_set.shape[0]
    num_batches_train = num_samples_train // batch_size
    num_samples_valid = test_set.shape[0]
    num_batches_valid = num_samples_valid // batch_size

    # setting up lists for handling loss/accuracy
    train_acc, train_loss = [], []
    valid_acc, valid_loss = [], []
    losses = []

    get_slice = lambda i, size: range(i * size, (i + 1) * size)

    for epoch in range(num_epochs):
        # Forward -> Backprob -> Update params
        # Train
        cur_loss = 0
        net.train()
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            # x_batch = Variable(torch.from_numpy(train_set[slce, :2]))
            x_batch = torch.from_numpy(train_set[slce, :2]).cuda()
            target = train_set[slce, 2]
            target_batch = torch.from_numpy(target).long().cuda()
            output = net(x_batch)

            # compute gradients given loss
            batch_loss = criterion(output, target_batch)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            cur_loss += batch_loss
        losses.append(cur_loss / batch_size)

        net.eval()
        # Evaluate training
        train_preds, train_targs = [], []
        for i in range(num_batches_train):
            slce = get_slice(i, batch_size)
            x_batch = Variable(torch.from_numpy(train_set[slce]))

            output = net(x_batch)
            preds = torch.max(output, 1)[1]

            train_targs += list(target)
            train_preds += list(preds.data.numpy())

        # Evaluate validation
        val_preds, val_targs = [], []
        for i in range(num_batches_valid):
            slce = get_slice(i, batch_size)
            x_batch = Variable(torch.from_numpy(test_set[slce]))

            output = net(x_batch)
            preds = torch.max(output, 1)[1]
            val_preds += list(preds.data.numpy())
            val_targs += list(target)

        train_acc_cur = accuracy_score(train_targs, train_preds)
        valid_acc_cur = accuracy_score(val_targs, val_preds)

        train_acc.append(train_acc_cur)
        valid_acc.append(valid_acc_cur)

        if epoch % 10 == 0:
            print("Epoch %2i : Train Loss %f , Train acc %f, Valid acc %f" % (
                epoch + 1, losses[-1], train_acc_cur, valid_acc_cur))

    epoch = np.arange(len(train_acc))
    plt.figure()
    plt.plot(epoch, train_acc, 'r', epoch, valid_acc, 'b')
    plt.legend(['Train Accucary', 'Validation Accuracy'])
    plt.xlabel('Updates'), plt.ylabel('Acc')
