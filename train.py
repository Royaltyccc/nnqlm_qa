import warnings
from torchtext.data import Field, TabularDataset, BucketIterator
from sklearn import metrics
from utils import *

warnings.filterwarnings("ignore")


def train_sim(opts):
    ID = Field(sequential=False, use_vocab=False, dtype=torch.float)
    Q_TEXT = Field(sequential=True,
                   use_vocab=True,
                   fix_length=opts.q_len,
                   lower=True,
                   tokenize=str.split,
                   batch_first=True)
    A_TEXT = Field(sequential=True,
                   use_vocab=True,
                   fix_length=opts.a_len,
                   lower=True, tokenize=str.split,
                   batch_first=True)
    TEXT = Field(sequential=True,
                 use_vocab=True,
                 fix_length=opts.a_len,
                 lower=True, tokenize=str.split,
                 batch_first=True)

    fields = [('question', Q_TEXT), ('ans_pos', A_TEXT), ('ans_neg', A_TEXT)]
    fields_test = [('qid', ID), ('aid', ID), ('question', Q_TEXT), ('answer', A_TEXT), ('label', ID)]
    fields_only_for_vocab = [('question', Q_TEXT), ('ans_pos', TEXT), ('ans_neg', TEXT)]

    train_only_for_vocab = TabularDataset(path=join('./data', opts.dataset_fn, 'train.tsv'), format='tsv',
                                          skip_header=True, fields=fields_only_for_vocab)

    dev, test = TabularDataset.splits(
        path=join('./data', opts.dataset_fn),
        format='tsv',
        validation='dev.tsv', test=opts.test + '.tsv',
        skip_header=True, fields=fields_test
    )

    # train by myself
    word_vectors = KeyedVectors.load(opts.embedding_fn, mmap='r')
    stoi = {word: idx for idx, word in enumerate(word_vectors.index2word)}
    i2v = [torch.from_numpy(word_vectors[word].copy()) for idx, word in enumerate(word_vectors.index2word)]
    # others
    # i2v, stoi = load_embedding(opts.embedding_fn)

    TEXT.build_vocab(train_only_for_vocab)
    TEXT.vocab.set_vectors(stoi, i2v, dim=opts.embedding_dim)

    Q_TEXT.vocab = TEXT.vocab
    A_TEXT.vocab = TEXT.vocab

    dev_iter, test_iter = BucketIterator.splits(
        (dev, test),
        batch_sizes=(opts.batch_size, opts.batch_size),
        device=torch.device(opts.device),
        sort=False)

    model = build_model_from_opts(opts, TEXT.vocab)
    optimizer = build_optimizer_from_opts(opts, model)

    best_map = 1
    test_per_k_epoch = opts.test_per_k_epoch
    for i in range(opts.epochs):
        train_loss_in_epoch = []

        dev_qid = []
        test_qid = []

        dev_label = []
        test_label = []

        dev_sim = []
        test_sim = []

        print("rebuilding train dataset for different negative answers")
        p = Processor("./data/insuranceQA/V1")
        p.get_train_data().to_csv(join(p.path_dir, 'train.tsv'), index=False, sep='\t')
        train = TabularDataset(path=join('./data/', opts.dataset_fn, 'train.tsv'), format='tsv',
                               skip_header=True, fields=fields)
        train_iter = BucketIterator(train, batch_size=opts.batch_size, device=opts.device, shuffle=True, sort=False)
        print("rebuilding train dataset done")

        for idx, batch in enumerate(tqdm(train_iter)):
            good_sim = model(batch.question, batch.ans_pos)
            bad_sim = model(batch.question, batch.ans_neg)
            train_loss = compute_loss(good_sim, bad_sim, opts)
            train_loss_in_epoch.append(train_loss)

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        train_loss = sum(train_loss_in_epoch) / len(train_loss_in_epoch)
        print("epoch :\t{}".format(i))
        print("train loss:\t{}".format(train_loss))

        if (i + 1) % test_per_k_epoch == 0:
            with torch.no_grad():
                for idx, batch in enumerate(tqdm(dev_iter)):
                    sim = model(batch.question, batch.answer)

                    dev_qid.extend(batch.qid.cpu().tolist())
                    dev_label.extend(batch.label.cpu().tolist())
                    dev_sim.extend(sim.cpu().tolist())

                for idx, batch in enumerate(tqdm(test_iter)):
                    sim = model(batch.question, batch.answer)

                    test_qid.extend(batch.qid.cpu().tolist())
                    test_label.extend(batch.label.cpu().tolist())
                    test_sim.extend(sim.cpu().tolist())

                dev_df = pd.DataFrame({"qid": dev_qid, "label": dev_label, "sim": dev_sim})
                test_df = pd.DataFrame({"qid": test_qid, "label": test_label, "sim": test_sim})

                dev_acc = calculate_acc(dev_df)
                test_acc = calculate_acc(test_df)
                dev_map, dev_mrr = calculate_map_mrr(dev_df)
                test_map, test_mrr = calculate_map_mrr(test_df)

                print("epoch :\t{}".format(i))
                print("train loss:\t{}".format(sum(train_loss_in_epoch) / len(train_loss_in_epoch)))
                print("-" * 50)
                print("dev_acc:\t{}\tdev_map:\t{}\tdev_mrr:\t{}".format(dev_acc, dev_map, dev_mrr))
                print("test_acc:\t{}\ttest_map:\t{}\ttest_mrr:\t{}".format(test_acc, test_map, test_mrr))

                if best_map > test_map:
                    best_map = test_map
                    save_path = join(opts.checkpoint_dir,
                                     '{}_{}_{}_map_{:.4f}'.format(opts.model_name,
                                                                  opts.dataset_fn.replace('/', '_'),
                                                                  i,
                                                                  best_map))
                    torch.save({
                        "opts": opts,
                        "model_state_dict": model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        "train_loss": train_loss,
                        "dev_acc": dev_acc,
                        "dev_map": dev_map,
                        "dev_mrr": dev_mrr,
                        "test_acc": test_acc,
                        "test_map": test_map,
                        "test_mrr": test_mrr

                    }, save_path)

                if test_per_k_epoch // 2 == 0:
                    test_per_k_epoch = 1
                else:
                    test_per_k_epoch = test_per_k_epoch // 2


def train_class(opts):
    Q_TEXT = Field(sequential=True,
                   use_vocab=True,
                   fix_length=opts.q_len,
                   lower=True,
                   tokenize=str.split,
                   batch_first=True)
    A_TEXT = Field(sequential=True,
                   use_vocab=True,
                   fix_length=opts.a_len,
                   lower=True, tokenize=str.split,
                   batch_first=True)

    TEXT = Field(sequential=True,
                 use_vocab=True,
                 fix_length=opts.q_len,
                 lower=True, tokenize=str.split,
                 batch_first=True)

    ID = Field(sequential=False, use_vocab=False, dtype=torch.float)
    LABEL = Field(sequential=False, use_vocab=False)

    fields = [('qid', ID), ('aid', ID), ('question', Q_TEXT), ('answer', A_TEXT), ('label', LABEL)]
    fields_only_for_vocab = [('qid', None), ('aid', None), ('question', TEXT), ('answer', TEXT), ('label', LABEL)]

    print("building dataset and iterator: starting")
    train, test = TabularDataset.splits(
        path=join('./data', opts.dataset_fn),
        format='tsv',
        train='train', test='test',
        skip_header=False, fields=fields
    )

    train_only_for_vocab, _ = TabularDataset.splits(
        path=join('./data', opts.dataset_fn),
        format='tsv',
        train='train', test='test',
        skip_header=False, fields=fields_only_for_vocab
    )

    train_iter, test_iter = BucketIterator.splits(
        (train, test),
        batch_sizes=(opts.batch_size, opts.batch_size),
        device=torch.device(opts.device),
        sort_key=lambda x: len(x.question),
        sort_within_batch=True
    )
    print("building dataset and iterator: done")

    print("building word vectors: starting")
    glove_vectors = KeyedVectors.load(opts.embedding_fn) if os.path.exists(opts.embedding_fn) \
        else gensim.downloader.load(opts.embedding_fn)
    stoi = {word: idx for idx, word in enumerate(glove_vectors.index2word)}
    i2v = [torch.from_numpy(glove_vectors[word].copy()) for idx, word in enumerate(glove_vectors.index2word)]

    TEXT.build_vocab(train_only_for_vocab)
    TEXT.vocab.set_vectors(stoi, i2v, dim=opts.embedding_dim,
                           unk_init=torch.Tensor.uniform_)  # Found no method to assign unk_init

    Q_TEXT.vocab = TEXT.vocab
    A_TEXT.vocab = TEXT.vocab
    print("building word vectors: done")

    model = build_model_from_opts(opts, TEXT.vocab)
    print(model)
    optimizer = build_optimizer_from_opts(opts, model)
    criterion = build_loss_from_opts(opts)

    best_map = 0
    for i in range(opts.epochs):
        train_loss_in_epoch = []
        test_loss_in_epoch = []

        train_qid = []
        train_aid = []
        train_label = []
        train_pred = []
        train_score = []

        test_qid = []
        test_aid = []
        test_label = []
        test_pred = []
        test_score = []

        model.train()
        is_first_batch = True
        for idx, batch in enumerate(tqdm(train_iter)):
            if idx < 300 and opts.dataset_fn == 'wiki' and opts.is_jump:
                continue
            elif idx < 3300 and opts.dataset_fn == 'trec' and opts.is_jump:
                continue
            if is_first_batch:
                first_batch = batch
                is_first_batch = False
                if first_batch.batch_size != opts.batch_size:
                    continue
            if batch.batch_size != opts.batch_size:
                batch = fill_last_batch(first_batch, batch)

            score = model(batch.question, batch.answer)
            pred = torch.argmax(score, dim=1)
            train_loss = criterion(score, batch.label)
            train_loss_in_epoch.append(train_loss.item())

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_qid.extend(batch.qid.cpu().tolist())
            train_aid.extend(batch.aid.cpu().tolist())
            train_label.extend(batch.label.cpu().tolist())
            train_pred.extend(pred.cpu().tolist())
            train_score.extend(score[:, 1].cpu().tolist())  # 标签为1的概率

        train_record = pd.DataFrame({"qid": train_qid,
                                     "aid": train_aid,
                                     "label": train_label,
                                     "score": train_score,
                                     "pred": train_pred})

        train_roc = metrics.roc_auc_score(train_record['label'], train_record['score'])
        train_map, train_mrr = calculate_map_mrr(train_record, sort_by='score')

        train_loss = sum(train_loss_in_epoch) / len(train_loss_in_epoch)
        # print("epoch {}:\ttrain_loss {}".format(i, train_loss))
        # print("train roc:{}\t train map:{}\t train mrr:{}\t".format(train_roc, train_map, train_mrr))

        model.eval()
        with torch.no_grad():
            if opts.is_jump:
                continue
            is_first_batch = True
            for idx, batch in enumerate(tqdm(test_iter)):
                if is_first_batch:
                    first_batch = batch
                    is_first_batch = False
                    if first_batch.batch_size != opts.batch_size:
                        continue
                if batch.batch_size != opts.batch_size:
                    batch = fill_last_batch(first_batch, batch)

                score = model(batch.question, batch.answer)
                pred = torch.argmax(score, dim=1)
                test_loss = criterion(score, batch.label)
                test_loss_in_epoch.append(test_loss.item())

                test_qid.extend(batch.qid.cpu().tolist())
                test_aid.extend(batch.aid.cpu().tolist())
                test_label.extend(batch.label.cpu().tolist())
                test_pred.extend(pred.cpu().tolist())
                test_score.extend(score[:, 1].cpu().tolist())

            test_record = pd.DataFrame({"qid": test_qid,
                                        "aid": test_aid,
                                        "label": test_label,
                                        "score": test_score,
                                        "pred": test_pred})

            test_roc = metrics.roc_auc_score(test_record['label'], test_record['score'])
            test_map, test_mrr = calculate_map_mrr(test_record, sort_by='score')

            test_loss = sum(test_loss_in_epoch) / len(test_loss_in_epoch)
            print("epoch: {} \t train loss:{} \t test loss:{} \n"
                  "\t \t \t train roc:{} \t test roc: {} \n"
                  "\t \t \t train mrr:{} \t test mrr:{}\n"
                  "\t \t \t train map:{} \t test map:{}".format(i,
                                                                train_loss,
                                                                test_loss,
                                                                train_roc,
                                                                test_roc,
                                                                train_mrr,
                                                                test_mrr,
                                                                train_map,
                                                                test_map))
            if test_map > best_map:
                best_map = test_map
                save_path = join(opts.checkpoint_dir,
                                 '{}_{}_{}_map_{:.4f}'.format(opts.model_name,
                                                              opts.dataset_fn.replace('/', '_'),
                                                              i,
                                                              best_map))
                torch.save({"opts": opts,
                            "model_state_dict": model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            "train_loss": train_loss,
                            "test_loss": test_loss,
                            "train_metric": [train_roc, train_map, train_mrr],
                            "test_metric": [test_roc, test_map, test_mrr]}, save_path)


if __name__ == '__main__':
    pass
