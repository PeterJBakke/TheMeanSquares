from types import SimpleNamespace
import torch
import random
from torch.autograd import Variable

from data import tokenizer

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def accuracy(output, target):
    correct_prediction = torch.abs(target - output)
    return torch.mean(correct_prediction.float())


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


def cite_u_like_train(dataset, train_iter, val_iter, net, test_iter, optimizer, criterion, num_user, text_stoi,
                      num_epochs=5):
    net.train()
    prev_epoch = 0
    train_loss, train_accs, train_length = [0, 0, 0]
    train_res = []
    val_res = []
    for batch in train_iter:
        # if train_iter.epoch != prev_epoch:
        if train_iter.iterations % 100 == 0:
            net.eval()
            val_loss, val_accs, val_length = [0, 0, 0]

            for val_batch in val_iter:
                val_input = load_text(dataset, val_batch, text_stoi)
                val_output = net(val_input).reshape(-1)
                val_target = Variable(torch.tensor([1 for _ in range(len(val_batch))]).float().to(device))
                val_loss += criterion(val_output, val_target) * val_batch.batch_size
                val_accs += accuracy_sigmoid(val_output, val_target) * val_batch.batch_size
                val_length += val_batch.batch_size

            val_loss /= val_length
            val_accs /= val_length
            val_res.append(val_accs)

            train_loss /= train_length
            train_accs /= train_length
            train_res.append(train_accs)
            print(
                "Epoch {}: Train loss: {:.2f}, Train avg error: {:.2f}, Validation loss: {:.2f}, Validation avg error: {:.2f}"
                    .format(train_iter.epoch, train_loss, train_accs, val_loss, val_accs))
            # plot_res(train_res, val_res, train_iter.epoch)

            net.train()

        net.train()

        positive_input = load_text(dataset, batch, text_stoi)
        pos_and_neg_batch = negative_sampling(positive_input, num_user)
        target_pos = Variable(torch.tensor([1 for _ in range(len(batch))]).to(device))
        target_neg = Variable(torch.tensor([0 for _ in range(len(batch))]).to(device))

        output = net(pos_and_neg_batch).reshape(-1)
        target = torch.cat((target_pos, target_neg), 0).float().to(device)
        batch_loss = criterion(output, target)
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        train_loss += criterion(output, target) * batch.batch_size * 2
        train_accs += accuracy_sigmoid(output, target) * batch.batch_size * 2
        train_length += batch.batch_size * 2

        if train_iter.iterations % 10 == 0:
            print(
                "Train loss: {:.3f}, Train avg error: {:.3f}"
                    .format(criterion(output, target), accuracy_sigmoid(output, target)))

        prev_epoch = train_iter.epoch
        if train_iter.epoch == num_epochs:
            break


def load_text(dataset, batch, text_stoi):
    texts = [get_text(dataset, doc_id, text_stoi) for doc_id in batch.doc.data.cpu().numpy()]
    longest_length = max([len(text) for text in texts])
    texts = torch.tensor(pad_list(texts, longest_length)).to(device)
    texts = torch.transpose(texts, 0, 1)

    batch_with_negative_sampling = {'user': batch.user, 'text': texts}
    return SimpleNamespace(**batch_with_negative_sampling)


def negative_sampling(batch, num_users):
    random_users = torch.tensor(
        [random.randint(0, num_users) for _ in range(len(batch.user))]
    ).to(device)

    user = torch.cat((batch.user, random_users), 0).to(device)
    text = torch.cat((batch.text, batch.text), 1).to(device)

    batch_with_negative_sampling = {'user': user, 'text': text}
    return SimpleNamespace(**batch_with_negative_sampling)


def accuracy_sigmoid(output, target):
    return torch.mean(torch.abs(output - target).float())


def print_params(net):
    for name, param in net.named_parameters():
        if param.requires_grad:
            print(name, param.data)


def get_text(dataset, doc_id, text_stoi):
    text = dataset.get_document_abstract(doc_id)
    tokens = tokenizer(text)
    return [text_stoi[token] for token in tokens]


def pad_list(text_list, length):
    return [text + ([1] * (length - len(text))) for text in text_list]
