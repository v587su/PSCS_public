import random
import torch
import tqdm
import pandas as pd
import os
import re
import random
from torch.utils.data import Dataset


class SiamessDataset(Dataset):
    def __init__(self, args, nl_corpus_path, nl_vocab, nl_seq_len, path_vocab,
                 path_seq_len, code_dir_path, k, is_train=True,
                 encoding="utf-8"):
        super(SiamessDataset, self).__init__()
        self.nl_vocab = nl_vocab
        self.path_vocab = path_vocab
        self.nl_seq_len = nl_seq_len
        self.path_seq_len = path_seq_len
        self.k = k
        self.nl_lines = None
        self.nl_corpus_path = nl_corpus_path
        self.encoding = encoding
        self.code = {}
        self.is_train = is_train

        def file_path(name, flag):
            if flag:
                return os.path.join(code_dir_path, '..', 'train', 'java',
                                    '{}.csv'.format(name))
            else:
                return os.path.join(code_dir_path, 'java',
                                    '{}.csv'.format(name))

        # path_contexts
        f = open(file_path('path_contexts', False), 'r')
        path_contexts = f.read()
        f.close()
        self.code['path_contexts'] = self.context_process(path_contexts)
        # node_types
        node_types = pd.read_csv(file_path('node_types', True), sep=',')
        self.code['node_types'] = self.df2vocab(node_types)
        # paths
        paths = pd.read_csv(file_path('paths', True), sep=',')
        self.code['paths'] = self.df2vocab(paths, max_len=self.path_seq_len,
                                           id_vocab=path_vocab)

        # tokens
        tokens = pd.read_csv(file_path('tokens', True), sep=',')
        self.code['tokens'] = self.df2vocab(tokens, id_vocab=nl_vocab)
        self.token_map = self.df2vocab(tokens[['id', 'token_split']],
                                       max_len=self.nl_seq_len,
                                       id_vocab=nl_vocab)

        with open(nl_corpus_path, "r", encoding=encoding) as f:
            self.nl_lines = [line[:-1] for line in
                             tqdm.tqdm(f, desc="Loading Dataset")]
        print(len(self.nl_lines))
        print(len(self.code['path_contexts']))
        print('===')

        self.nl_tokenized = self.tokenize(self.nl_lines, self.nl_vocab,
                                          self.nl_seq_len)
        self.corpus_length = len(self.code['path_contexts'])

    def __len__(self):
        return self.corpus_length

    def __getitem__(self, item):
        data = self.get_data(item)
        if self.is_train:
            trg_item = self.get_random_item(item)
            trg_data = self.get_data(trg_item)
            return data, trg_data
        else:
            return data

    def tokenize(self, corpus, vocab, seq_len):
        rtn = []
        for line in corpus:
            tokens = line.split()
            for i in range(len(tokens)):
                tokens[i] = vocab.stoi.get(tokens[i], vocab.unk_index)
            value_list = [vocab.sos_index] + tokens + [vocab.eos_index]
            value_list = value_list[:seq_len]
            padding = [vocab.pad_index for _ in
                       range(seq_len - len(value_list))]
            value_list.extend(padding)
            rtn.append(value_list)
        return rtn

    def get_random_item(self, item):
        rand_item = item
        while rand_item is item:
            rand_item = random.randrange(self.corpus_length)
        return rand_item

    def get_data(self, item):
        code_id = self.code['path_contexts'][item][0]
        path_context = self.code['path_contexts'][item][1]

        nl = self.nl_tokenized[code_id]
        start_tokens = []
        paths = []
        end_tokens = []

        for i in random.sample(path_context, self.k):
            start_token, path, end_token = i.split(',')
            start_tokens.append(
                self.token_map.to_str(int(start_token)))
            end_tokens.append(
                self.token_map.to_str(int(end_token)))
            paths.append(
                self.code['paths'].to_str(int(path), is_path=True))

        data = {
            "nl": torch.tensor(nl),
            "paths": torch.tensor(paths),
            "start_tokens": torch.tensor(start_tokens),
            "end_tokens": torch.tensor(end_tokens),
        }

        return data

    def df2vocab(self, d, max_len=None, id_vocab=None):
        return Vocab({int(row[0]): row[1] for i, row in d.iterrows()},
                     max_len, id_vocab=id_vocab)

    def context_process(self, context):
        context = context.split('\n')
        rtn = []
        match_num = re.compile(r'(?<=_)[0-9]+(?=\.)')
        for i, row in enumerate(context[:-1]):
            row = row.split(' ')
            id = int(re.search(match_num, row[0]).group())
            values = row[1:]
            if len(values) < self.k:
                padding = ['0,0,0' for _ in
                           range(self.k - len(values))]
                values.extend(padding)
            rtn.append((id, values))
        return rtn


class Vocab:
    def __init__(self, my_dict, seq_max_len=None, pad_index=0,
                 id_vocab=None):
        self.pad_index = pad_index
        self.unknow_token = 1
        self.has_id = id_vocab is not None
        if seq_max_len is not None:
            self.seq_max_len = seq_max_len
            for key, value in my_dict.items():
                value_list = [int(a) for a in value.split()]
                if self.has_id:
                    value_list = [id_vocab.sos_index] + value_list + [
                        id_vocab.eos_index]
                    value_list = value_list[:seq_max_len]
                    padding = [id_vocab.pad_index for _ in
                               range(seq_max_len - len(value_list))]
                    value_list.extend(padding)
                    my_dict[key] = value_list
                else:
                    padding = [self.pad_index for _ in
                               range(seq_max_len - len(value_list))]
                    value_list.extend(padding)
                    my_dict[key] = value_list
        self.itos = my_dict
        if isinstance(my_dict[1], str) and not self.has_id:
            self.stoi = {value: key for key, value in my_dict.items()}

    def to_id(self, sequence, unk=1):
        if isinstance(sequence, list):
            return [self.stoi.get(s, unk) for s in sequence]
        else:
            return self.stoi.get(sequence, unk)

    def to_str(self, sequence, is_path=False):
        unknown = [self.unknow_token] + [0 for _ in range(self.seq_max_len - 1)] \
            if is_path | self.has_id else self.unknow_token
        if isinstance(sequence, list):
            return [self.itos.get(i, unknown)
                    for i in sequence]
        else:
            return self.itos.get(sequence, unknown)
