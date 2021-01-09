import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


def compute_loss(good_s, bad_s, opts):
    return torch.relu(opts.margin - good_s + bad_s).sum()


def outer_product(x):
    return torch.einsum('bli, blj->blij', x, x)


def last_pool(t):
    return t[:, -1, :, :].view(t.size()[0], -1)


def max_pool(t):
    return torch.max(t, dim=1)[0].view(t.size()[0], -1)


def mean_pool(t):
    return torch.mean(t, dim=1).view(t.size()[0], -1)


def get_qa_by_mode(q, a, mode):
    if mode == 'max':
        return max_pool(q), max_pool(a)
    elif mode == 'mean':
        return mean_pool(q), mean_pool(a)
    elif mode == 'last':
        return last_pool(q), last_pool(a)


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
    def __init__(self, vocab, embedding_dim, n_layers, hidden_size, dropout_lstm, dropout_linear, is_bi, mode='max'):
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
        self.is_bi = is_bi

        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.embeddings.weight.data.copy_(vocab.vectors)
        if is_bi:
            self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=True, dropout=dropout_lstm,
                                bidirectional=True)
            self.linear = nn.Sequential(nn.Linear(self.hidden_size * 4, 2),
                                        nn.Dropout(dropout_linear))
        else:
            self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, batch_first=True, dropout=dropout_lstm)
            self.linear = nn.Sequential(nn.Linear(self.hidden_size * 2, 2),
                                        nn.Dropout(dropout_linear))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, q, a):
        # (B, L)
        q = self.embeddings(q)  # (B, L, D)
        a = self.embeddings(a)  # (B, L, D)
        q_o, _ = self.lstm(q)  # (B, L, 2*H) | (B, L, 4*H)
        a_o, _ = self.lstm(a)  # (B, L, 2*H) | (B, L, 4*H)

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


class QaRnnForClass(nn.Module):
    def __init__(self, vocab, embedding_dim, n_layers, hidden_size, dropout_lstm, dropout_linear, is_bi, mode='max'):
        super().__init__()
        self.mode = mode
        self.hidden_size = hidden_size
        self.is_bi = is_bi

        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.embeddings.weight.data.copy_(vocab.vectors)
        if is_bi:
            self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True, dropout=dropout_lstm,
                              bidirectional=True)
            self.linear = nn.Sequential(nn.Linear(self.hidden_size * 4, 2),
                                        nn.Dropout(dropout_linear))
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_size, n_layers, batch_first=True, dropout=dropout_lstm)
            self.linear = nn.Sequential(nn.Linear(self.hidden_size * 2, 2),
                                        nn.Dropout(dropout_linear))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, q, a):
        # (B, L)
        q = self.embeddings(q)  # (B, L, D)
        a = self.embeddings(a)  # (B, L, D)
        q_o, _ = self.rnn(q)  # (B, L, H) | (B, L, 2*H)
        a_o, _ = self.rnn(a)  # (B, L, H) | (B, L, 2*H)

        q, a = get_qa_by_mode(q_o, a_o, mode=self.mode)  # (B, H) | (B, 2*H)
        x = torch.cat((q, a), dim=1)  # (B, 2*H) | (B, 4*H)
        x = self.linear(x)  # (B, 2)
        x = self.logsoftmax(x)
        return x


class DensityLayer(nn.Module):
    '''
    input shape: (B, L, D)
    B: batch_size
    L: sentence length
    D: word_embedding dimension

    output shape: (B, L, D, D)
    '''

    def __init__(self, batch_size, sen_len, word_dim):
        super().__init__()
        self.batch_size = batch_size
        self.sentence_length = sen_len
        self.word_dim = word_dim
        self.W = Parameter(torch.ones(size=(self.batch_size, self.sentence_length, self.word_dim, self.word_dim)),
                           requires_grad=True)  # (B, L, D, D)
        nn.init.uniform(self.W)

    def forward(self, x):
        '''
        :param self:
        :param x: input shape: (B, L, D) batch_size, sentence_length, dimension
        :return: output shape: (B, L, D, D)
        '''

        def dot(x):
            '''
            compute input's dot product
            :param x: shape: (B, L, D)
            :return: shape: (B, L, 1)
            '''
            return torch.matmul(x.unsqueeze(2), x.unsqueeze(3)).squeeze(3)

        x = outer_product(x) / (dot(x).view(self.batch_size, self.sentence_length, 1, 1) + 1e-4)
        return x


class CnnBasedRnnCell(nn.Module):
    def __init__(self, num_filter, conv_ks, padding):
        '''
        the gate's input should be concatenated with h(B, 1, D, D) and x(B, 1, D, D) ->input(B, 1, 2*D, D)
        stride must be (2, 1) to avoid change the hidden_size. then the shape of h generated by input will be (B, 1, D, D)
        :param num_filter:
        :param conv_ks: along with padding to make sure the hidden_size == word_dim
        :param padding:
        '''
        super().__init__()
        self.num_filter = num_filter
        self.new_h = nn.Sequential(nn.Conv2d(in_channels=1,
                                             out_channels=num_filter,
                                             kernel_size=conv_ks,
                                             stride=(2, 1),
                                             padding=padding),
                                   nn.Tanh())

    def forward(self, x, pre_h):
        '''
        :param x: shape: (B, 1, D, D)
        :param pre_h: shape: (B, 1, H, H) == (B, 1, D, D)
        :param pre_c: shape: (B, 1, H, H) == (B, 1, D, D)
        :return:
            o == h: (B, 1, D, D)
            c: (B, 1, D, D)
        '''
        input = torch.cat((x, pre_h), dim=2)  # (B, 1, 2*D, D)
        h = self.new_h(input)  # (B, F, D, D)

        return h


class CnnBasedLstmCell(nn.Module):
    def __init__(self, num_filter, conv_ks, padding):
        '''
        the gate's input should be concatenated with h(B, 1, D, D) and x(B, 1, D, D) ->input(B, 1, 2*D, D)
        stride must be (2, 1) to avoid change the hidden_size. then the shape of h generated by input will be (B, 1, D, D)
        :param num_filter:
        :param conv_ks: along with padding to make sure the hidden_size == word_dim
        :param padding:
        '''
        super().__init__()
        self.num_filter = num_filter
        self.forget_gate = nn.Sequential(nn.Conv2d(in_channels=1,
                                                   out_channels=num_filter,
                                                   kernel_size=conv_ks,
                                                   stride=(2, 1),
                                                   padding=padding),
                                         nn.Sigmoid())
        self.input_gate = nn.Sequential(nn.Conv2d(in_channels=1,
                                                  out_channels=num_filter,
                                                  kernel_size=conv_ks,
                                                  stride=(2, 1),
                                                  padding=padding),
                                        nn.Sigmoid())
        self.output_gate = nn.Sequential(nn.Conv2d(in_channels=1,
                                                   out_channels=num_filter,
                                                   kernel_size=conv_ks,
                                                   stride=(2, 1),
                                                   padding=padding),
                                         nn.Sigmoid())
        self.candidate_cell = nn.Sequential(nn.Conv2d(in_channels=1,
                                                      out_channels=num_filter,
                                                      kernel_size=conv_ks,
                                                      stride=(2, 1),
                                                      padding=padding),
                                            nn.Tanh())

        # self.linear_c = nn.Linear(self.num_filter, 1)
        # self.linear_h = nn.Linear(self.num_filter, 1)

    def forward(self, x, pre_h, pre_c):
        '''
        :param x: shape: (B, 1, D, D)
        :param pre_h: shape: (B, 1, H, H) == (B, 1, D, D)
        :param pre_c: shape: (B, 1, H, H) == (B, 1, D, D)
        :return:
            o == h: (B, 1, D, D)
            c: (B, 1, D, D)
        '''
        input = torch.cat((x, pre_h), dim=2)  # (B, 1, 2*D, D)
        fg = self.forget_gate(input)  # (B, F, D, D)
        ig = self.input_gate(input)
        og = self.output_gate(input)
        cs = self.candidate_cell(input)

        c = fg * pre_c + ig * cs  # (B, F, D, D)
        h = og * torch.tanh(c)

        return h, c


class CnnBasedRnn(nn.Module):
    def __init__(self, num_filter, num_layer, conv_ks, padding, dropout_lstm):
        super().__init__()
        self.num_layers = num_layer
        self.cbrcs = nn.ModuleList([CnnBasedRnnCell(num_filter, conv_ks, padding) for _ in range(num_layer)])
        self.dropout = nn.Dropout(dropout_lstm)

    def forward(self, x):
        '''
        :param x: shape: (B, L, D, D)
        :return:
        '''
        pre_x = x
        seq_len = x.size()[1]

        for idx, cbrc in enumerate(self.cbrcs):
            ht = torch.zeros_like(pre_x)[:, 0, :, :].unsqueeze(dim=1)  # （B, 1, D, D）
            all_h = ht  # (B, 1, D, D) -> (B, T, D, D) -> (B, L, D, D)
            for t in range(seq_len):
                ht = cbrc(pre_x[:, t, :, :].unsqueeze(dim=1), ht)  # (B, 1, D, D)
                all_h = torch.cat((all_h, ht), dim=1)
            if idx < self.num_layers - 1 and self.dropout != 0:
                all_h = self.dropout(all_h)
            all_h = all_h[:, 1:, :, :]  # Remove h0
            pre_x = all_h
        return all_h


class CnnBasedLstm(nn.Module):
    def __init__(self, num_filter, num_layer, conv_ks, padding, dropout_lstm):
        super().__init__()
        self.num_layers = num_layer
        self.cblcs = nn.ModuleList([CnnBasedLstmCell(num_filter, conv_ks, padding) for _ in range(num_layer)])
        self.dropout = nn.Dropout(dropout_lstm)

    def forward(self, x):
        '''
        :param x: shape: (B, L, D, D)
        :return:
        '''
        pre_x = x
        seq_len = x.size()[1]

        for idx, cblc in enumerate(self.cblcs):
            ht = torch.zeros_like(pre_x)[:, 0, :, :].unsqueeze(dim=1)  # （B, 1, D, D）
            ct = torch.zeros_like(pre_x)[:, 0, :, :].unsqueeze(dim=1)
            all_h = ht  # (B, 1, D, D) -> (B, T, D, D) -> (B, L, D, D)
            all_c = ct
            for t in range(seq_len):
                ht, ct = cblc(pre_x[:, t, :, :].unsqueeze(dim=1), ht, ct)  # (B, 1, D, D)
                all_h = torch.cat((all_h, ht), dim=1)
                all_c = torch.cat((all_c, ct), dim=1)
            if idx < self.num_layers - 1 and self.dropout != 0:
                all_h = self.dropout(all_h)
            all_h = all_h[:, 1:, :, :]  # Remove h0
            all_c = all_c[:, 1:, :, :]
            pre_x = all_h
        return all_h, (all_h, all_c)


class NnqlmCnnBasedRNN(nn.Module):
    def __init__(self, vocab, embedding_dim, batch_size, q_len, a_len, num_layer, num_filter, conv_ks,
                 padding, dropout_lstm, dropout_linear, is_bi=False, mode='max'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = embedding_dim
        self.mode = mode
        self.is_bi = is_bi

        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.embeddings.weight.data.copy_(vocab.vectors)
        self.q_density = DensityLayer(batch_size, q_len, embedding_dim)
        self.a_density = DensityLayer(batch_size, a_len, embedding_dim)
        self.cbr_forward = CnnBasedRnn(num_filter, num_layer, conv_ks, padding, dropout_lstm)
        self.cbr_backward = CnnBasedRnn(num_filter, num_layer, conv_ks, padding, dropout_lstm)
        if self.is_bi:
            self.hidden2label_score = nn.Sequential(nn.Linear(4 * self.hidden_size * self.hidden_size, 2),
                                                    nn.Dropout(dropout_linear))
        else:
            self.hidden2label_score = nn.Sequential(nn.Linear(2 * self.hidden_size * self.hidden_size, 2),
                                                    nn.Dropout(dropout_linear))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, q, a):
        bs = q.size()[0]
        sl = q.size()[1]
        di = self.embedding_dim
        q_e = self.embeddings(q)
        a_e = self.embeddings(a)
        q_d = self.q_density(q_e)  # (B, L, D, D)
        a_d = self.a_density(a_e)
        # q_o, _ = self.q_cbl(q)  # (B, L, D, D)
        # a_o, _ = self.a_cbl(a)

        qf_o = self.cbr_forward(q_d)  # (B, L, D*D)
        af_o = self.cbr_forward(a_d)
        qf_o = qf_o.view(bs, sl, -1)
        af_o = af_o.view(bs, sl, -1)

        if self.is_bi:
            qb_o = self.cbr_backward(q_d.flip(1, ))  # Reverse seq order (B, L, D*D)
            ab_o = self.cbr_backward(a_d.flip(1, ))
            qb_o = qb_o.view(bs, sl, -1)
            ab_o = ab_o.view(bs, sl, -1)

            q = torch.empty(bs, sl, 2 * di * di).to(device=q.device)  # (B, L, 2*D*D)
            a = torch.empty(bs, sl, 2 * di * di).to(device=a.device)  # (B, L, 2*D*D)
            q[:, :, ::2] = qf_o
            q[:, :, 1::2] = qb_o
            a[:, :, ::2] = af_o
            a[:, :, 1::2] = ab_o

        else:
            q = qf_o
            a = af_o

        # (B, L, D*D) -> (B, D*D) | (B, L, 2*D*D) -> (B, 2*D*D)
        q, a = get_qa_by_mode(q, a, self.mode)

        qa = torch.cat((q, a), dim=1)  # (B, 2*D*D) / (B, 4*D*D)
        score = self.hidden2label_score(qa)
        score = self.logsoftmax(score)
        return score


class NnqlmCnnBasedLstm(nn.Module):
    def __init__(self, vocab, embedding_dim, batch_size, q_len, a_len, num_layer, num_filter, conv_ks,
                 padding, dropout_lstm, dropout_linear, is_bi=False, mode='max'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = embedding_dim
        self.mode = mode
        self.is_bi = is_bi

        self.embeddings = nn.Embedding(len(vocab), embedding_dim)
        self.embeddings.weight.data.copy_(vocab.vectors)
        self.q_density = DensityLayer(batch_size, q_len, embedding_dim)
        self.a_density = DensityLayer(batch_size, a_len, embedding_dim)
        self.cbl_forward = CnnBasedLstm(num_filter, num_layer, conv_ks, padding, dropout_lstm)
        self.cbl_backward = CnnBasedLstm(num_filter, num_layer, conv_ks, padding, dropout_lstm)
        if self.is_bi:
            self.hidden2label_score = nn.Sequential(nn.Linear(4 * self.hidden_size * self.hidden_size, 2),
                                                    nn.Dropout(dropout_linear))
        else:
            self.hidden2label_score = nn.Sequential(nn.Linear(2 * self.hidden_size * self.hidden_size, 2),
                                                    nn.Dropout(dropout_linear))
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, q, a):
        '''
        Here H(hidden_size) have to be equaled to D(word_dim) for the reason of concatenating

        q|a -> density : (B, L, D) -> (B, L, D, D)

        single step in CNN-LSTM:
            gate: density -> conv2d : (B, D, D) -> (B, D, D)
            lstm single output :
                    ht : (B, H, H) == (B, D, D)
                    ct : (B, H, H) == (B, D, D)

        all output in CNN-LSTM: output, (ht, ct)
            output: (B, L, H, H) == (B, L, D, D)
        :param q:
        :param a:
        :return:
        '''
        bs = q.size()[0]
        sl = q.size()[1]
        di = self.embedding_dim
        q_e = self.embeddings(q)
        a_e = self.embeddings(a)
        q_d = self.q_density(q_e)  # (B, L, D, D)
        a_d = self.a_density(a_e)
        # q_o, _ = self.q_cbl(q)  # (B, L, D, D)
        # a_o, _ = self.a_cbl(a)

        qf_o, _ = self.cbl_forward(q_d)  # (B, L, D*D)
        af_o, _ = self.cbl_forward(a_d)
        qf_o = qf_o.view(bs, sl, -1)
        af_o = af_o.view(bs, sl, -1)

        if self.is_bi:
            qb_o, _ = self.cbl_backward(q_d.flip(1, ))  # Reverse seq order (B, L, D*D)
            ab_o, _ = self.cbl_backward(a_d.flip(1, ))
            qb_o = qb_o.view(bs, sl, -1)
            ab_o = ab_o.view(bs, sl, -1)

            q = torch.empty(bs, sl, 2 * di * di).to(device=q.device)  # (B, L, 2*D*D)
            a = torch.empty(bs, sl, 2 * di * di).to(device=a.device)  # (B, L, 2*D*D)
            q[:, :, ::2] = qf_o
            q[:, :, 1::2] = qb_o
            a[:, :, ::2] = af_o
            a[:, :, 1::2] = ab_o

        else:
            q = qf_o
            a = af_o

        # (B, L, D*D) -> (B, D*D) | (B, L, 2*D*D) -> (B, 2*D*D)
        q, a = get_qa_by_mode(q, a, self.mode)

        qa = torch.cat((q, a), dim=1)  # (B, 2*D*D) / (B, 4*D*D)
        score = self.hidden2label_score(qa)
        score = self.logsoftmax(score)
        return score
