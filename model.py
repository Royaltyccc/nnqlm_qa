import torch
import torch.nn as nn


def compute_loss(good_s, bad_s, opts):
    return torch.relu(opts.margin - good_s + bad_s).sum()


class QaLstmForSim(nn.Module):
    def __init__(self, vocab, embedding_dim, n_layers, hidden_size, dropout, mode='max'):
        '''
        :param vocab:
        :param embedding_dim:
        :param n_layers:
        :param hidden_size:
        :param dropout:
        :param mode: max|mean|last
        '''
        super().__init__()
        self.mode = mode
        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.embeddings.weight.data.copy_(vocab.vectors)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=True, dropout=dropout,
                              bidirectional=True)
        self.cos = nn.CosineSimilarity(dim=1)

    def forward(self, q, a):
        # (B, L)
        q = self.embeddings(q)  # (B, L, D)
        a = self.embeddings(a)  # (B, L, D)
        q_o, _ = self.bilstm(q)  # (B, L, 2*H)
        a_o, _ = self.bilstm(a)  # (B, L, 2*H)

        if self.mode == 'max':
            q = torch.max(q_o, dim=1)[0]
            a = torch.max(a_o, dim=1)[0]
        elif self.mode == 'last':
            q = q_o[:, -1, :]
            a = a_o[:, -1, :]
        elif self.mode == 'mean':
            q = torch.mean(q_o, dim=1)
            a = torch.mean(a_o, dim=1)

        sim = self.cos(q, a)
        return sim


class QaLstmForClass(nn.Module):
    def __init__(self, vocab, embedding_dim, n_layers, hidden_size, dropout, mode='max'):
        '''
        :param vocab: pytorch vocab class
        :param embedding_dim:
        :param n_layers:
        :param hidden_size:
        :param dropout:
        :param mode: max|mean|last
        '''
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.embeddings.weight.data.copy_(vocab.vectors)
        self.bilstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=True, dropout=dropout,
                              bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 4, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, q, a):
        # (B, L)
        q = self.embeddings(q)  # (B, L, D)
        a = self.embeddings(a)  # (B, L, D)
        q_o, _ = self.bilstm(q)  # (B, L, 2*H)
        a_o, _ = self.bilstm(a)  # (B, L, 2*H)

        # (B, 2*H)
        if self.mode == 'max':
            q = torch.max(q_o, dim=1)[0]
            a = torch.max(a_o, dim=1)[0]
        elif self.mode == 'last':
            q = q_o[:, -1, :]
            a = a_o[:, -1, :]
        elif self.mode == 'mean':
            q = torch.mean(q_o, dim=1)
            a = torch.mean(a_o, dim=1)

        x = torch.cat((q, a), dim=1)  # (B, 4*H)
        x = self.linear(x)  # (B, 2)
        x = self.logsoftmax(x)
        return x
