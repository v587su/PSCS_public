import os
import jsonlines
import re
import pickle
import tqdm
from collections import Counter



def read_jsonl_file(dir_path):
    dir_name = ['test', 'train', 'valid']
    # dir_name = ['test']
    # jsons = {'test': []}
    jsons = {'test': [], 'train': [], 'valid': []}
    for type in dir_name:
        files = os.listdir(os.path.join(dir_path, type))
        for file_name in files:
            if file_name == '.DS_Store':
                continue
            print(file_name)
            f = open(os.path.join(dir_path, type, file_name), 'r+',
                     encoding='utf8')
            for line in jsonlines.Reader(f):
                jsons[type].append(line)
            f.close()

    return jsons['test'], jsons['train'], jsons['valid']
    # return jsons['test']


def split_camel(camel_str,test=''):
    try:
        split_str = re.sub(
            r'(?<=[a-z]|[0-9])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\s+',
            '_',
            camel_str)
    except TypeError:
        return ['']
    try:
        if split_str[0] == '_':
            return [camel_str]
    except IndexError:
        return []
    return split_str.lower().split('_')

