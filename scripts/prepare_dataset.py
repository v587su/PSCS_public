import pandas as pd
import argparse
from dataset import WordVocab

from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dir_path", required=True, type=str)
    parser.add_argument("-o", "--output_dir_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    args = parser.parse_args()

    node_path = os.path.join(args.dir_path, 'node_types.csv')
    corpus_path = os.path.join(args.dir_path, 'paths.csv')
    contexts_path = os.path.join(args.dir_path, 'path_contexts.csv')

    nodes_vocab = pd.read_csv(node_path)
    nodes_vocab['node_type'] = nodes_vocab.apply(
        lambda x: '_'.join(x['node_type'].split()), axis=1)
    node_dict = nodes_vocab.set_index('id').to_dict(orient='dict')
    node_dict = node_dict['node_type']
    paths = pd.read_csv(corpus_path)
    paths = paths.apply(
        lambda x: ' '.join(
            [node_dict.get(int(i), '<unk>') for i in x['path'].split(' ')]),
        axis=1)
    path_list = paths.values.tolist()

    vocab = WordVocab(path_list, max_size=args.vocab_size,
                      min_freq=args.min_freq)
    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(
        os.path.join(args.output_dir_path, 'path_vocab.pickle'))

    # 用bert_vocab处理tokens里的subtoken
    f = open(os.path.join(args.output_dir_path, 'nl_vocab.pickle'), 'rb')
    nl_vocab = pickle.load(f)
    f.close()

    def process_tokens(x):
        split_list = split_camel(x['token'], x)
        split_list = nl_vocab.to_seq(split_list)
        return ' '.join([str(i) for i in split_list])


    tokens_paths = [os.path.join(args.dir_path, 'tokens.csv'),
                    os.path.join(args.dir_path, '..', '..', 'test', 'java',
                                 'tokens.csv')]

    for tokens_path in tokens_paths:
        tokens = pd.read_csv(tokens_path)
        tokens['token_split'] = tokens.apply(process_tokens, axis=1)
        tokens['token_split'] = tokens['token_split'].astype(str)
        tokens.to_csv(tokens_path, index=False)
