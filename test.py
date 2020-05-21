import pickle
import torch
import argparse
import numpy as np
import math
from main import args_parser
from model import *
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataset import SiamessDataset, WordVocab


def test(args):
    model_save_dir = '{}/storage/exp9/main.model.ep{}'.format(args.save_dir, str(args.test_epoch))

    f = open('{}/processed/nl_vocab.pickle'.format(args.data_dir), 'rb')
    nl_vocab = pickle.load(f)
    f.close()

    f = open('{}/processed/path_vocab.pickle'.format(args.data_dir), 'rb')
    path_vocab = pickle.load(f)
    f.close()
    corpus_test = SiamessDataset(args,
                                 '{}/processed/siamess_nl_test.txt'.format(
                                     args.data_dir),
                                 nl_vocab, args.nl_seq_len, path_vocab,
                                 args.path_len,
                                 '{}/path_data/test'.format(args.data_dir),
                                 args.k, is_train=False)
    model = torch.load(model_save_dir)
    cuda_condition = args.with_cuda == 1
    if cuda_condition:
        model.cuda()
    acc, mrr, map, ndcg = eval(args, corpus_test, model,
                               1000, args.topK)
    post_fix = {
        "acc": acc,
        "mrr": mrr,
        "map": map,
        "ndcg": ndcg
    }
    print(str(post_fix))


def eval(args, valid_set, model, poolsize, K):
    """
    validation in full test set.
    """

    def ACC(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1: sum = sum + 1
        return sum / float(len(real))

    def MAP(real, predict):
        sum = 0.0
        for id, val in enumerate(real):
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1: sum = sum + (id + 1) / float(index + 1)
        return sum / float(len(real))

    def MRR(real, predict):
        sum = 0.0
        for val in real:
            try:
                index = predict.index(val)
            except ValueError:
                index = -1
            if index != -1: sum = sum + 1.0 / float(index + 1)
        return sum / float(len(real))

    def NDCG(real, predict):
        dcg = 0.0
        idcg = IDCG(len(real))
        for i, predictItem in enumerate(predict):
            if predictItem in real:
                itemRelevance = 1
                rank = i + 1
                dcg += (math.pow(2, itemRelevance) - 1.0) * (
                        math.log(2) / math.log(rank + 1))
        return dcg / float(idcg)

    def IDCG(n):
        idcg = 0
        itemRelevance = 1
        for i in range(n): idcg += (math.pow(2, itemRelevance) - 1.0) * (
                math.log(2) / math.log(i + 2))
        return idcg

    model.eval()

    device = next(model.parameters()).device

    data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                              batch_size=poolsize,
                                              shuffle=True, drop_last=False,
                                              num_workers=1)
    accs, mrrs, maps, ndcgs = [], [], [], []
    all_nl_emb = []
    all_code_vec = []
    n = 0
    for data in data_loader:
        n += 1
        if args.with_cuda:
            torch.cuda.empty_cache()
        with torch.no_grad():
            for key, value in data.items():
                data[key] = value.to(device)

            nl_emb, code_vec = model.forward(data['nl'],
                                             data["paths"],
                                             data["start_tokens"],
                                             data["end_tokens"])

            all_nl_emb.append(nl_emb)
            all_code_vec.append(code_vec)

    all_nl_emb = torch.cat(all_nl_emb)
    all_code_vec = torch.cat(all_code_vec)
    data_num = all_nl_emb.size(0)
    for i in range(data_num):
        nl_vec_rep = all_nl_emb[i].view(1, -1).expand(data_num, -1)
        sims = F.cosine_similarity(
            all_code_vec, nl_vec_rep).data.cpu().numpy()
        negsims = np.negative(sims)
        predict = np.argsort(negsims)
        predict = predict[:K]
        predict = [int(k) for k in predict]
        real = [i]
        accs.append(ACC(real, predict))
        mrrs.append(MRR(real, predict))
        maps.append(MAP(real, predict))
        ndcgs.append(NDCG(real, predict))

    return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)


if __name__ == '__main__':
    args = args_parser()
    test(args)
