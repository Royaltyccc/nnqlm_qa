import hashlib
import random
import os
from os.path import join as join

import pandas as pd
from tqdm import tqdm
from model import *
from sklearn.utils import shuffle
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader


def get_vocabulary(path):
    idx2vocab = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            idx, vocab = line.split()
            idx2vocab[idx] = vocab
    return idx2vocab


class Processor:
    def __init__(self, path_dir):
        super().__init__()
        self.path_dir = path_dir
        self.idx2vocab = get_vocabulary(join(self.path_dir, "vocabulary"))  # string->string
        self.label2answer = self.get_answer(join(self.path_dir, "answers.label.token_idx"),
                                            self.idx2vocab)  # int->string

        self.path_train = join(self.path_dir, "question.train.token_idx.label")
        self.path_dev = join(self.path_dir, "question.dev.label.token_idx.pool")
        self.path_test1 = join(self.path_dir, "question.test1.label.token_idx.pool")
        self.path_test2 = join(self.path_dir, "question.test2.label.token_idx.pool")

    def hash_str(self, s):
        return int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % (10 ** 8)

    def convert_raw_to_tsv(self):

        train_data = self.get_train_data()
        train_data.to_csv(join(self.path_dir, 'train.tsv'), index=False, sep='\t')
        dev_data = self.get_test_data(self.path_dev)
        dev_data.to_csv(join(self.path_dir, 'dev.tsv'), index=False, sep='\t')
        test1_data = self.get_test_data(self.path_test1)
        test1_data.to_csv(join(self.path_dir, 'test1.tsv'), index=False, sep='\t')
        test2_data = self.get_test_data(self.path_test2)
        test2_data.to_csv(join(self.path_dir, 'test2.tsv'), index=False, sep='\t')

    def get_answer(self, path, vocab):
        label2answer = {}
        with open(path, 'r') as f:
            for line in f.readlines():
                label, idx = line.split('\t')
                label2answer[int(label)] = ' '.join([vocab[i] for i in idx.split()])
        return label2answer

    def get_train_data(self):
        # f in format idx1, idx2, ... \t label
        questions = []
        answers_pos = []
        answers_neg = []
        with open(self.path_train, 'r') as f:
            for line in tqdm(f.readlines()):
                idx, label = line.split('\t')
                question = ' '.join([self.idx2vocab[i] for i in idx.split()])
                ans_pos = [self.label2answer[int(lb)] for lb in label.split()]
                ans_neg = random.sample(list(self.label2answer.values()), len(ans_pos))

                questions.extend([question] * len(ans_pos))
                answers_pos.extend(ans_pos)
                answers_neg.extend(ans_neg)

                assert len(questions) == len(answers_pos) == len(answers_neg)
        return pd.DataFrame({"question": questions, "ans_pos": answers_pos, "ans_neg": answers_neg})

    def get_test_data(self, path):
        questions = []
        answers = []
        q_id = []
        a_id = []
        is_pos = []
        with open(path, 'r') as f:
            for line in tqdm(f.readlines()):
                ans_pos_label, vocab_idx, ans_neg_label = line.split('\t')
                question = ' '.join([self.idx2vocab[idx] for idx in vocab_idx.split()])
                ans_pos = [self.label2answer[int(lb)] for lb in ans_pos_label.split()]
                ans_neg = [self.label2answer[int(lb)] for lb in ans_neg_label.split()]

                questions.extend([question] * len(ans_pos + ans_neg))
                answers.extend(ans_pos + ans_neg)

                q_id.extend([self.hash_str(question)] * len(ans_pos + ans_neg))
                a_id.extend([self.hash_str(a) for a in ans_pos + ans_neg])
                is_pos.extend([1] * len(ans_pos))
                is_pos.extend([0] * len(ans_neg))

                assert len(questions) == len(answers) == len(is_pos)
        return pd.DataFrame({"qid": q_id, "aid": a_id, "question": questions, "answer": answers, "label": is_pos})

    def get_all_sentence(self):
        train_data = self.get_train_data()
        test_data = pd.concat((self.get_test_data(self.path_dev),
                               self.get_test_data(self.path_test1),
                               self.get_test_data(self.path_test2)))
        test_sentence = list(set(test_data['question'].tolist()))

        train_sentence = list(set(train_data['question'].tolist())) + list(self.label2answer.values())
        return train_sentence + test_sentence

    def train_wv(self):
        sentences = [s.lower().split() for s in self.get_all_sentence()]
        word_num = len(set(w for s in sentences for w in s))
        model = Word2Vec(sentences=sentences, size=100, window=5, workers=36, min_count=1, iter=50)
        word_vectors = model.wv
        word_vectors.save(join(self.path_dir, 'word2vec_100_dim'))


def build_model_from_opts(opts, vocab):
    if opts.model_name == 'qa-lstm-sim':
        model = QaLstmForSim(vocab,
                             opts.embedding_dim,
                             opts.n_layers,
                             opts.hidden_size,
                             opts.dropout,
                             opts.model_mode).to(device=opts.device)
    elif opts.model_name == 'qa-lstm-cls':
        model = QaLstmForClass(vocab,
                               opts.embedding_dim,
                               opts.n_layers,
                               opts.hidden_size,
                               opts.dropout,
                               opts.model_mode).to(device=opts.device)
    elif opts.model_name == 'qa-nnqlm-cnnlstm':
        model = NnqlmCnnBasedLstm(vocab,
                                  opts.embedding_dim,
                                  opts.batch_size,
                                  opts.q_len,
                                  opts.a_len,
                                  opts.embedding_dim,
                                  opts.n_filter,
                                  opts.filter_size,
                                  opts.padding,
                                  opts.model_mode).to(device=opts.device)
    if opts.is_continue:
        fns = os.listdir(opts.checkpoint_dir)
    return model


def build_optimizer_from_opts(opts, model):
    if opts.optimizer == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=opts.learning_rate)
    elif opts.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=opts.learning_rate)
    elif opts.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=opts.learning_rate)
    elif opts.optimizer == 'adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=opts.learning_rate)


def build_loss_from_opts(opts):
    if opts.loss == 'nll':
        return nn.NLLLoss()


def calculate_map_mrr_0(dataframe):
    def get_ap_and_rr(df):
        n_good = int(df['gal'].tolist()[0])
        max_r = df['sim'].idxmax()
        max_n = df['sim'][:n_good].idxmax()
        rank = df['rank']
        ap = 1 if max_r == max_n else 0
        rr = 1 / float(rank[max_r] - rank[max_n] + 1)
        return ap, rr

    ap_sum = 0
    rr_sum = 0
    df_gs = dataframe.groupby(dataframe["qid"])
    dataframe['rank'] = df_gs.rank(method='max')['sim']
    for name, group in df_gs:
        ap, rr = get_ap_and_rr(group)
        ap_sum += ap
        rr_sum += rr
    return ap_sum / len(df_gs), rr_sum / len(df_gs)


def calculate_map_mrr(dataframe, sort_by='sim'):
    def mrr_metric(group):
        group = shuffle(group, random_state=121)
        candidates = group.sort_values(by=sort_by, ascending=False).reset_index()
        rr = candidates[candidates["label"] == 1].index.min() + 1
        if rr != rr:
            return 0.
        return 1.0 / rr

    def map_metric(group):
        group = shuffle(group, random_state=121)
        ap = 0
        candidates = group.sort_values(by=sort_by, ascending=False).reset_index()
        correct_candidates = candidates[candidates["label"] == 1]
        correct_candidates_index = candidates[candidates["label"] == 1].index
        if len(correct_candidates) == 0:
            return 0
        for i, index in enumerate(correct_candidates_index):
            ap += 1.0 * (i + 1) / (index + 1)
        return ap / len(correct_candidates)

    ap_sum = 0
    rr_sum = 0
    df_gs = dataframe.groupby(dataframe["qid"])
    dataframe['rank'] = df_gs.rank(ascending=False, method='max')[sort_by]
    for name, group in df_gs:
        ap_sum += map_metric(group)
        rr_sum += mrr_metric(group)
    return ap_sum / len(df_gs), rr_sum / len(df_gs)


def calculate_acc(dataframe):
    acc_sum = 0
    df_gs = dataframe.groupby(dataframe["qid"])
    for name, group in df_gs:
        sim_max = group['sim'].max()
        candidate_g = group[group['sim'] == sim_max]
        if 1 in list(candidate_g['label']):
            acc_sum += 1
    return acc_sum / len(df_gs)


def fill_last_batch(first_batch, cur_batch):
    '''
    batch attribute
    qid, aid (B,)
    question answer (B, L)
    label (B,)
    '''
    to_fill_len = first_batch.batch_size - cur_batch.batch_size
    filled_batch = first_batch

    filled_batch.qid = torch.cat((cur_batch.qid, first_batch.qid[:to_fill_len]), dim=0)
    filled_batch.aid = torch.cat((cur_batch.aid, first_batch.aid[:to_fill_len]), dim=0)
    filled_batch.label = torch.cat((cur_batch.label, first_batch.label[:to_fill_len]), dim=0)
    filled_batch.question = torch.cat((cur_batch.question, first_batch.question[:to_fill_len, :]), dim=0)
    filled_batch.answer = torch.cat((cur_batch.answer, first_batch.answer[:to_fill_len, :]), dim=0)

    return filled_batch


def load_embedding(filename):
    embeddings = list()
    word2idx = dict()
    print('start loading embedding')
    with open(filename, mode='r', encoding='utf-8') as f:
        for line in f:
            arr = line.strip().split(' ')
            embedding = [float(val) for val in arr[1:len(arr)]]
            word2idx[arr[0].lower()] = len(word2idx)
            embeddings.append(torch.Tensor(embedding))

    embedding_size = len(arr) - 1
    word2idx['UNKNOWN'] = len(word2idx)
    embeddings.append([0] * embedding_size)

    print('embedding loaded')
    return embeddings, word2idx


if __name__ == '__main__':
    path_dir = './data/insuranceQA/V1'
    processor = Processor(path_dir)
    # processor.convert_raw_to_tsv()
    # processor.train_wv()

    word_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
    word_vectors.save('./data/glove-wiki-gigaword-100')
    # word_vectors = KeyedVectors.load(join(path_dir, 'word2vec_100_dim'), mmap='r')
    # stoi = {word: idx for idx, word in enumerate(word_vectors.index2word)}
    # i2v = [torch.from_numpy(word_vectors[word].copy()) for idx, word in enumerate(word_vectors.index2word)]
    # print(torch.from_numpy(word_vectors['hard'].copy()))

    # temp = pd.read_csv('./data/temp.tsv', sep='\t')
    # df_gs = temp.groupby(temp["qid"])
    # temp['rank_o'] = df_gs.rank(ascending=False, method='max')['sim']
    # temp.to_csv('./data/temp.tsv', index=False, sep='\t')
    # acc = calculate_acc(temp)
    # print(acc)

    pass
