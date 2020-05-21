import os
import re
import pandas as pd
import collections
from utils import *


def split_token(row, sub_token_dict):
    sub_token_list = split_camel(row['token'])
    # for sub_token in sub_token_list:
    #     sub_token_dict[sub_token] += 1
    return ' '.join(sub_token_list)


def generate_sub_token(row, vocab):
    ids = []
    for sub_token in row['token_split'].split(' '):
        ids.append(str(vocab.get(sub_token, 1)))
    # row['token'] = ' '.join(ids)
    return ' '.join(ids)


if __name__ == '__main__':
    dir_names = ['train', 'test', 'valid']
    crt_path = os.path.dirname(os.path.abspath(__file__))

    for dir_name in dir_names:
        data_dir_path = os.path.join(crt_path, '..', 'data','path_data', dir_name,
                                     'java')
        tokens = pd.read_csv(
            os.path.join(data_dir_path, 'tokens.csv'))
        sub_tokens = collections.defaultdict(lambda: 0)
        tokens['token_split'] = tokens.apply(lambda x: split_token(x, sub_tokens), axis=1)
        sub_token_vocab = {key: i + 1 for i, (key, item) in
                           enumerate(sub_tokens.items())}
        tokens['token_split'] = tokens.apply(
            lambda x: generate_sub_token(x, sub_token_vocab),
            axis=1)
        tokens['token_split'] = tokens['token_split'].astype(str)
        sub_token_vocab = [[item, key] for key, item in
                           sub_token_vocab.items()]
        sub_token_vocab = pd.DataFrame(sub_token_vocab,
                                       columns=['id', 'sub_token'])
        tokens.to_csv(os.path.join(data_dir_path, 'tokens.csv'), index=False)
        sub_token_vocab.to_csv(os.path.join(data_dir_path, 'sub_tokens.csv'),
                               index=False)
