import torch
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn


# Depending on arg, build dataset
def get_model(embeddings, args):
    print("\nBuilding model...")
    if args.bow or args.tfidf:
        return LinearBoWClassifier(args.vocab_size)
    elif args.model_name == 'dan':
        return DAN(embeddings, args)
    elif args.model_name == 'FC':
        return DAN(embeddings, args)
    elif args.model_name == 'rnn':
        return RNN(embeddings, args)
    elif args.model_name == 'lstm1':
        return LSTM1(embeddings, args)
    elif args.model_name == 'lstm2':
        return LSTM2(embeddings, args)
    else:
        raise Exception("Model name {} not supported!".format(args.model_name))


class DAN(nn.Module):

    def __init__(self, embeddings, args):
        super(DAN, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
        # self.embedding_layer = nn.Embedding( vocab_size, embed_dim)
        # self.embedding_layer.weight.data = torch.from_numpy( embeddings )
        self.W_hidden = nn.Linear(embed_dim, args.num_hidden)
        self.W_out = nn.Linear(args.num_hidden, 1)

    def forward(self, x_indx):
        all_x = self.embedding_layer(x_indx)
        avg_x = torch.mean(all_x, dim=1)
        hidden = F.relu(self.W_hidden(avg_x))
        out = self.W_out(hidden)
        return out

class FC(nn.Module):
    def __init__(self, embeddings, args):
        super(FC, self).__init__()
        self.args = args
        _, embed_dim = embeddings.shape
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
        self.fc1 = nn.Linear(embed_dim*100, args.num_hidden)
        # self.fc2 = nn.Linear(args.num_hidden_discriminator, args.num_hidden_discriminator)
        self.fc3 = nn.Linear(args.num_hidden, 1)

    def forward(self, x):
        xf = x.view(-1, x.size()[1]*x.size()[2])
        h = F.elu(self.fc1(xf))
        # h = F.elu(self.fc2(h))
        out = self.fc3(h)
        return out


class RNN(nn.Module):

    def __init__(self, embeddings, args):
        super(RNN, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
        # self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        # self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=args.num_hidden,
                          num_layers=1, batch_first=True)
        self.W_o = nn.Linear(args.num_hidden, 1)

    def forward(self, x_indx):
        all_x = self.embedding_layer(x_indx)
        h0 = autograd.Variable(torch.randn(1, self.args.batch_size, args.num_hidden))
        output, h_n = self.rnn(all_x, h0)
        h_n = h_n.squeeze(0)
        out = self.W_o(h_n)
        return out


class LSTM1(nn.Module):

    def __init__(self, embeddings, args):
        super(LSTM1, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embed_dim = embed_dim
        # self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        # self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=args.num_hidden,
                          num_layers=1, batch_first=True)
        self.W_o = nn.Linear(args.num_hidden, 1)

    def init_hidden_states(self, batch_size):
        h0 = autograd.Variable(torch.randn(2, batch_size, self.args.num_hidden // 2))
        c0 = autograd.Variable(torch.randn(2, batch_size, self.args.num_hidden // 2))
        if self.args.cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, x_indx):
        batch_size = x.size()[0]
        all_x = self.embedding_layer(x_indx)
        h0, c0 = self.init_hidden_states(batch_size)
        output, h_n, c_n = self.lstm(all_x, (h0, c0))
        h_n = h_n.squeeze(0)
        out = self.W_o(h_n)
        return out

class LSTM2(nn.Module):

    def __init__(self, embeddings, args):
        super(LSTM2, self).__init__()
        self.args = args
        vocab_size, embed_dim = embeddings.shape
        self.embed_dim = embed_dim
        self.embedding_layer = nn.Embedding.from_pretrained(torch.from_numpy(embeddings), freeze=True)
        # self.embedding_layer.weight.data = torch.from_numpy(embeddings)
        # self.embedding_layer.weight.requires_grad = False
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=(args.num_hidden // 2),
                          num_layers=1, batch_first=True, bidirectional=True, dropout=0.1)
        self.fc1 = nn.Linear(args.num_hidden, 1)

    def init_hidden_states(self, batch_size):
        h0 = autograd.Variable(torch.randn(2, batch_size, self.args.num_hidden // 2))
        c0 = autograd.Variable(torch.randn(2, batch_size, self.args.num_hidden // 2))
        if self.args.cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()
        return (h0, c0)

    def forward(self, x_indx):
        batch_size = x_indx.size()[0]
        all_x = self.embedding_layer(x_indx)
        h0, c0 = self.init_hidden_states(batch_size)
        _, (h_n, c_n) = self.lstm(all_x, (h0, c0))
        h_n = h_n.view(-1, self.args.num_hidden)
        out = self.fc1(h_n)
        return out


class LinearBoWClassifier(nn.Module):  # inheriting from nn.Module!

    def __init__(self, vocab_size):
        super(LinearBoWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, 1)
        # self.W_hidden = nn.Linear(vocab_size, 200)
        # self.W_out = nn.Linear(200, 1)

    def forward(self, bow_vec):
        # out = F.log_softmax(self.linear(bow_vec), dim=1)
        out = self.linear(bow_vec)
        # hidden = F.sigmoid(self.W_hidden(bow_vec))
        # out = self.W_out(hidden)
        return out
