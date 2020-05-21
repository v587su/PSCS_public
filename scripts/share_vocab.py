# 后面token要切分，test不能直接unknown
import os
import pandas as pd
import tqdm
import argparse

max_row = 0


def trans(value, dict, unknown=0):
    new_value = dict.get(int(value), unknown)

    if pd.isna(new_value):
        new_value = unknown

    return str(int(new_value))


def transform_paths(row, map_dict):
    paths_ids = row['path'].split()
    new_ids = []
    for id in paths_ids:
        new_id = trans(id, map_dict)
        new_ids.append(new_id)
    new_path_id = ' '.join(new_ids)
    row['path'] = new_path_id
    return row


def transform_paths_content(row, token_map, path_map):
    row = row.split(',')
    start_token_id = row[0]
    path_id = row[1]
    end_token_id = row[2]
    new_start_token_id = trans(start_token_id, token_map)
    new_path_id = trans(path_id, path_map)
    new_end_token_id = trans(end_token_id, token_map)
    return '{},{},{}'.format(new_start_token_id, new_path_id,
                             new_end_token_id)


def fill_nan(vocab_map):
    global max_row
    max_row = vocab_map['id_y'].max()

    def apply_new_id(row):
        global max_row
        if pd.isna(row['id_y']):
            row['id_y'] = int(max_row + 1)
            max_row = row['id_y']
        return row

    vocab_map = vocab_map.apply(apply_new_id, axis=1)

    return vocab_map


def vocab_merge(vocab_a, vocab_b, on, method):
    vocab = vocab_a.merge(vocab_b, on=on, how=method)
    if method == 'outer':
        vocab = fill_nan(vocab)
    return vocab


def save_vocab(vocab, path, columns=None):
    vocab = vocab.iloc[:, 1:]
    if columns is not None:
        vocab.columns = columns
    try:
        vocab = vocab[[columns[1], columns[0]]].astype({'id': 'int32'})
    except ValueError:
        print(vocab)
    vocab.to_csv(path, index=False)


def map2dict(vocab_map):
    map_dict = {}
    for i, row in vocab_map.iterrows():
        if pd.isna(row[0]):
            continue
        map_dict[int(row[0])] = row[2]
    return map_dict


if __name__ == '__main__':
    # read files
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_dataset_dir", type=str,
                        help="train dataset")
    parser.add_argument("-v", "--test_dataset_dir", type=str,
                        help="test dataset")
    parser.add_argument("-o", "--output_dir", type=str,
                        help="output dir")
    parser.add_argument("--merge_vocab", type=bool, default=False,
                        help="merge vocab")
    args = parser.parse_args()
    print(args.train_dataset_dir)
    print(args.output_dir)

    cur_path = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(cur_path, '..', args.train_dataset_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    test_dir = os.path.join(cur_path, '..', args.test_dataset_dir)
    output_dir = os.path.join(cur_path, '..', args.output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    need_merge = args.merge_vocab
    node_vocab_test = pd.read_csv(os.path.join(test_dir, 'node_types.csv'))
    node_vocab_train = pd.read_csv(os.path.join(train_dir, 'node_types.csv'))
    token_vocab_test = pd.read_csv(os.path.join(test_dir, 'tokens.csv'))
    token_vocab_train = pd.read_csv(os.path.join(train_dir, 'tokens.csv'))
    path_vocab_test = pd.read_csv(os.path.join(test_dir, 'paths.csv'))
    path_vocab_train = pd.read_csv(os.path.join(train_dir, 'paths.csv'))

    # make path map
    method = 'outer' if need_merge else 'left'
    node_vocab_map = vocab_merge(node_vocab_test, node_vocab_train,
                                 on=['node_type'], method=method)
    token_vocab_map = vocab_merge(token_vocab_test, token_vocab_train,
                                  on=['token'], method='outer')

    node_dict = map2dict(node_vocab_map)
    token_dict = map2dict(token_vocab_map)

    path_vocab_test = path_vocab_test.apply(
        lambda row: transform_paths(row, node_dict), axis=1)

    path_vocab_map = vocab_merge(path_vocab_test, path_vocab_train,
                                 on=['path'],
                                 method='outer')
    # print('node:', node_vocab_map)
    # print('token:', token_vocab_map)
    # print('path:', path_vocab_map)
    path_dict = map2dict(path_vocab_map)
    # transform path_context
    path_context_test = []
    for root, dirs, files in os.walk(test_dir):
        for f_name in tqdm.tqdm(files):
            if 'path_contexts' in f_name:
                f_path = os.path.join(root, f_name)
                with open(f_path) as f:
                    f_list = f.readlines()
                for row in f_list:
                    path_list = row.split()
                    id = path_list[0]
                    paths = path_list[1:]
                    new_paths = []
                    for path_item in paths:
                        new_path = transform_paths_content(path_item,
                                                           token_dict,
                                                           path_dict)
                        new_paths.append(new_path)
                    new_row = ' '.join([str(id)] + new_paths) + '\n'
                    path_context_test.append(new_row)
    if need_merge:
        path_context_train = []
        for root, dirs, files in os.walk(train_dir):
            for f_name in tqdm.tqdm(files):
                if 'path_contexts' in f_name:
                    f_path = os.path.join(root, f_name)
                    with open(f_path) as f:
                        f_list = f.readlines()
                    path_context_train = path_context_train + f_list
        path_context_train = path_context_test + path_context_train
        f = open(os.path.join(output_dir, 'path_contexts.csv'), 'w')
        f.write(''.join(path_context_train))
        f.close()
        save_vocab(node_vocab_map, os.path.join(output_dir, 'node_types.csv'),
                   columns=['node_type', 'id'])
        save_vocab(token_vocab_map, os.path.join(output_dir, 'tokens.csv'),
                   columns=['token', 'id'])
        save_vocab(path_vocab_map, os.path.join(output_dir, 'paths.csv'),
                   columns=['path', 'id'])
    else:
        f = open(os.path.join(output_dir, 'path_contexts.csv'), 'w')
        f.write(''.join(path_context_test))
        f.close()
        save_vocab(path_vocab_map, os.path.join(train_dir, 'paths.csv'),
                   columns=['path', 'id'])
        save_vocab(token_vocab_map, os.path.join(train_dir, 'tokens.csv'),
                   columns=['token', 'id'])
        # node_type\paths\tokens 全用train的, path_contexts用test的
