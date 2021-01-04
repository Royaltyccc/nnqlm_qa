import argparse
from train import train_sim
from train import train_class


def get_parser():
    basic_parser = argparse.ArgumentParser(add_help=False)
    basic_parser.add_argument('--dataset_fn', type=str)
    basic_parser.add_argument('--epochs', type=int, default=10)
    basic_parser.add_argument('--test_per_k_epoch', type=int, default=10)
    basic_parser.add_argument('--batch_size', type=int, default=20)
    basic_parser.add_argument('--q_len', type=int, default=200)
    basic_parser.add_argument('--a_len', type=int, default=200)
    basic_parser.add_argument('--test', type=str, default='test1')
    basic_parser.add_argument('--embedding_fn', type=str, default='./data/glove-wiki-gigaword-50')
    basic_parser.add_argument('--vocab_fn', type=str, default='./data/insuranceQA/V1/vocabulary')
    basic_parser.add_argument('--embedding_dim', type=int, default=100)
    basic_parser.add_argument('--model_name', type=str, default='qa-lstm')
    basic_parser.add_argument('--model_mode', type=str, default='max')
    basic_parser.add_argument('--optimizer', type=str, default='sgd')
    basic_parser.add_argument('--loss', type=str, default='nll')
    basic_parser.add_argument('--learning_rate', type=float, default=0.01)
    basic_parser.add_argument('--dropout_lstm', type=float, default=0.4)
    basic_parser.add_argument('--dropout_linear', type=float, default=0.4)
    basic_parser.add_argument('--dropout_cnn', type=float, default=0.4)
    basic_parser.add_argument('--margin', type=float, default=0.2)

    basic_parser.add_argument('--device', type=str, default="cuda:0")
    basic_parser.add_argument('--checkpoint_dir', type=str, default="./checkpoints")
    basic_parser.add_argument('--is_continue', action='store_true', default=False)

    # lstm_parser = argparse.ArgumentParser(parents=[basic_parser])
    basic_parser.add_argument('--hidden_size', type=int, default=128)
    basic_parser.add_argument('--n_layer', type=int, default=1)
    basic_parser.add_argument('--is_bi_directional', action='store_true', default=False)

    # cnn_parser = argparse.ArgumentParser(parents=[basic_parser])
    basic_parser.add_argument('--filter_size', type=int, default=3)
    basic_parser.add_argument('--n_filter', type=int, default=5)
    basic_parser.add_argument('--padding', type=int, default=1)

    basic_parser.add_argument('--is_jump', action='store_true', default=False)
    return basic_parser


if __name__ == '__main__':
    basic_parser = get_parser()
    opts = basic_parser.parse_args()
    print(opts)
    if opts.dataset_fn == 'trec' or opts.dataset_fn == 'wiki':
        train_class(opts)
    else:
        train_sim(opts)
