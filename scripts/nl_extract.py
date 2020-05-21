import argparse
from dataset import WordVocab
from utils import *
from tqdm import tqdm


def str2file(data, prefix, type, k=None):
    if k is not None:
        if len(data) > k:
            for i in range(0, len(data), k):
                data_piece = data[i:i + k]
                save_file(data_piece, prefix, '{}_{}'.format(type, str(i)),
                          start_point=i)
        else:
            save_file(data, prefix, '{}_{}'.format(type, '0'), start_point=0)
    else:
        save_file(data, prefix, type)


def save_file(data, prefix, type, start_point=0):
    for i, snippet in tqdm(enumerate(data)):
        code_str = snippet['code']
        # pathminer v0.3 do not support single function
        code_str = "package nothing; class Hi {%s}" % code_str
        cur_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(cur_path, '..', 'data', 'path_files', type)
        if not os.path.exists(
                os.path.join(cur_path, '..', 'data', 'path_files')):
            os.mkdir(os.path.join(cur_path, '..', 'data', 'path_files'))
        if not os.path.exists(file_path):
            os.mkdir(file_path)
        f = open(os.path.join(file_path, '{}_{}.java'.format(prefix, str(
            i + start_point))), 'w', encoding='utf8')
        f.write(code_str)
        f.close()


def get_first_sentence(docstring):
    if '.' in docstring:
        docstring = docstring.split('.')[0]
    elif '\n' in docstring:
        docstring = docstring.split('\n')[0]
    return docstring


def str_filter(docstring):
    if re.search(r'[\u4e00-\u9fa5@]', docstring) is not None:
        return ''
    docstring = docstring.replace('\n', ' ')
    # remove words in (){}[]<>
    docstring = re.sub(r'\\(.*?\\)|\\{.*?}|\\[.*?]|\\<.*?>', '', docstring)
    # remove special tokens
    docstring = docstring.replace('_', ' ')
    docstring = re.sub(r'[^a-zA-Z0-9 ]', '', docstring)
    docstring_tokens = split_camel(docstring)
    docstring = ' '.join(docstring_tokens) if len(
        docstring_tokens) > 2 else ' '
    return docstring


def make_corpus(jsonl_list):
    corpus = []
    length_count = []
    new_jsonl_list = []
    for line in jsonl_list:
        first_sentence = get_first_sentence(line['docstring'])
        docstring = str_filter(first_sentence)
        if len(docstring) < 2:
            continue
        corpus.append(docstring)
        length_count.append(len(re.findall(r' ', docstring)) + 1)
        new_jsonl_list.append(line)

    print(len(corpus))
    corpus = '\n'.join(corpus)

    return corpus, new_jsonl_list


def save_corpus(file_content, file_name):
    current_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(
            os.path.join(current_path, '..', 'data', 'processed')):
        os.mkdir(os.path.join(current_path, '..', 'data', 'processed'))
    f = open(os.path.join(current_path, '..', 'data', 'processed', file_name),
             'w', encoding='utf8')
    f.write(file_content)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--k', type=int)

    args = parser.parse_args()

    current_path = os.path.dirname(os.path.abspath(__file__))
    test, train, valid = read_jsonl_file(
        os.path.join(current_path, '..', 'data', 'java'))

    siamess_nl_train, train = make_corpus(train)
    siamess_nl_test, test = make_corpus(test)
    siamess_nl_valid, valid = make_corpus(valid)
    print('train len:', len(train))
    print('test len:', len(test))
    print('valid len:', len(valid))
    save_corpus(siamess_nl_train, 'siamess_nl_train.txt')
    save_corpus(siamess_nl_test, 'siamess_nl_test.txt')
    save_corpus(siamess_nl_valid, 'siamess_nl_valid.txt')
    str2file(train, 'train', 'train', k=args.k)
    str2file(test, 'test', 'test')
    str2file(valid, 'valid', 'valid')

    vocab = WordVocab(siamess_nl_train.split('\n'), max_size=50000,
                      min_freq=1)

    print("VOCAB SIZE:", len(vocab))
    current_path = os.path.dirname(os.path.abspath(__file__))
    vocab.save_vocab(os.path.join(current_path, '..', 'data', 'processed',
                                  'nl_vocab.pickle'))
