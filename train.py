import torch
import random


def accuracy(output, target):
    correct_prediction = torch.abs(target - output)
    return torch.mean(correct_prediction.float())


def train(train_iter, val_iter, net, test_iter, optimizer, criterion, num_epochs=5):
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


def negative_sampling(batch, num_user, device):
    random_users = [random.randint(1, num_user) for _ in range(len(batch))]
    rating = [0 for _ in range(len(batch))]

    user_tensor = torch.tensor(random_users).to(device)
    rating_tensor = torch.tensor(rating).to(device)
    batch.user = user_tensor
    batch.rating = rating_tensor
    return batch
