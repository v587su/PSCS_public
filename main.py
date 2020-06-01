import pickle
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import math
import numpy as np
import os
import random
from torch.optim import Adam
from model import *
from dataset import SiamessDataset, WordVocab
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm

seed = 231
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


def train(args):
    f = open('{}/processed/nl_vocab.pickle'.format(args.data_dir), 'rb')
    nl_vocab = pickle.load(f)
    f.close()
    f = open('{}/processed/path_vocab.pickle'.format(args.data_dir), 'rb')
    path_vocab = pickle.load(f)
    f.close()

    print('cuda:', args.with_cuda)
    cuda_condition = args.with_cuda == 1
    device = torch.device("cuda:0" if cuda_condition else "cpu")
    corpus_train = SiamessDataset(args,
                                  '{}/processed/siamess_nl_train.txt'.format(
                                      args.data_dir),
                                  nl_vocab, args.nl_seq_len, path_vocab,
                                  args.path_len,
                                  '{}/path_data/train'.format(args.data_dir),
                                  args.k, is_train=True)
    corpus_test = SiamessDataset(args,
                                 '{}/processed/siamess_nl_test.txt'.format(
                                     args.data_dir),
                                 nl_vocab, args.nl_seq_len, path_vocab,
                                 args.path_len,
                                 '{}/path_data/test'.format(args.data_dir),
                                 args.k, is_train=False)

    train_data_loader = DataLoader(corpus_train, batch_size=args.batch_size)

    print('token vocab len', len(corpus_train.code['tokens'].itos))
    print('node vocab len', len(corpus_train.code['node_types'].itos))

    model = PSCSNetwork(args, len(nl_vocab), len(path_vocab), device)
    print("Total Parameters:",
          sum([p.nelement() for p in model.parameters()]))

    loss = RankLoss(args)
    optim = Adam(model.parameters(), lr=args.lr,
                 betas=[args.adam_beta1, args.adam_beta2],
                 weight_decay=args.adam_weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda
        epoch: args.decay_ratio ** epoch)

    use_parallel = False
    if cuda_condition:
        if torch.cuda.device_count() > 1:
            use_parallel = True
            model = nn.DataParallel(model)
        model = model.to(device)

    for epoch in range(args.epochs):
        data_iter = tqdm.tqdm(enumerate(train_data_loader),
                              total=len(train_data_loader),
                              bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        model.train()
        for i, (data, trg_data) in data_iter:
            for key, value in data.items():
                data[key] = value.to(device)
            for key, value in trg_data.items():
                trg_data[key] = value.to(device)
            if use_parallel:
                nl_vec, code_vec = model.module.forward(
                    data['nl'],
                    data["paths"],
                    data["start_tokens"],
                    data["end_tokens"])

                trg_nl_vec, trg_code_vec = model.module.forward(
                    trg_data['nl'],
                    trg_data["paths"],
                    trg_data["start_tokens"],
                    trg_data["end_tokens"])
            else:
                nl_vec, code_vec = model.forward(data['nl'],
                                                 data["paths"],
                                                 data["start_tokens"],
                                                 data["end_tokens"])

                trg_nl_vec, trg_code_vec = model.forward(
                    trg_data['nl'],
                    trg_data["paths"],
                    trg_data["start_tokens"],
                    trg_data["end_tokens"])

            loss_value = loss(nl_vec, trg_nl_vec, code_vec, trg_code_vec)
            optim.zero_grad()
            loss_value.backward()
            if args.clip_grad_norm is not None:
                clip_grad_norm(model.parameters(), args.clip_grad_norm)
            optim.step()
            avg_loss += float(loss_value.item())

            if i % args.log_freq == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss_value.item()
                }
                data_iter.write(str(post_fix))

            if i % args.val_freq == 0:
                acc, mrr, map, ndcg = validate(args, corpus_test, model,
                                               args.pool_size, args.topK)
                post_fix = {
                    "acc": acc,
                    "mrr": mrr,
                    "map": map,
                    "ndcg": ndcg
                }
                data_iter.write(str(post_fix))
                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)
                f = open('{}/performance.txt'.format(args.save_dir), 'a')
                post_fix['epoch'] = epoch
                f.write(str(post_fix) + '\n')
                f.close()

        scheduler.step()
        model_save_dir = '{}/main_model'.format(args.save_dir)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        output_path = model_save_dir + "/main.model.ep%d" % epoch
        torch.save(model.cpu(), output_path)
        model.to(device)


def validate(args, valid_set, model, poolsize, K):
    """
    simple validation in a code pool.
    @param: poolsize - size of the code pool, if -1, load the whole test set
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
                                              shuffle=True, drop_last=True,
                                              num_workers=1)
    accs, mrrs, maps, ndcgs = [], [], [], []
    for data in tqdm.tqdm(data_loader):
        if args.with_cuda:
            torch.cuda.empty_cache()
        with torch.no_grad():
            for key, value in data.items():
                data[key] = value.to(device)

            nl_emb, code_vec = model.forward(data['nl'],
                                             data["paths"],
                                             data["start_tokens"],
                                             data["end_tokens"])
        for i in range(poolsize):
            nl_vec_rep = nl_emb[i].view(1, -1).expand(poolsize, -1)
            n_results = K
            sims = F.cosine_similarity(
                code_vec, nl_vec_rep).data.cpu().numpy()
            negsims = np.negative(sims)
            predict = np.argsort(negsims)
            predict = predict[:n_results]
            predict = [int(k) for k in predict]
            real = [i]
            accs.append(ACC(real, predict))
            mrrs.append(MRR(real, predict))
            maps.append(MAP(real, predict))
            ndcgs.append(NDCG(real, predict))
    model.train()
    return np.mean(accs), np.mean(mrrs), np.mean(maps), np.mean(ndcgs)


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="output")

    parser.add_argument("-k", "--k", type=int, default=40,
                        help="k paths at a time")

    parser.add_argument("-nls", "--nl_seq_len", type=int, default=20,
                        help="maximum sequence len of natural language")
    parser.add_argument("-pl", "--path_len", type=int, default=12)
    parser.add_argument("-es", "--emb_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("-dp", "--dropout", type=float, default=0.25)
    parser.add_argument("-rdp", "--rnn_dropout", type=float, default=0.5)
    parser.add_argument("-m", "--margin", type=float, default=1,
                        help="margin")
    parser.add_argument("--pool_size", type=int, default=1000)
    parser.add_argument("--topK", type=int, default=10)

    parser.add_argument("-b", "--batch_size", type=int, default=64,
                        help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0,
                        help="dataloader worker size")

    parser.add_argument('-cuda', "--with_cuda", type=int, default=1,
                        help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="printing loss every n iter: setting n")
    parser.add_argument("--val_freq", type=int, default=10000,
                        help="validate every n iter: setting n")
    parser.add_argument("--test_epoch", type=int, default=0,
                        help="the epoch of trained model to test")

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="learning rate of adam")
    parser.add_argument("--decay_ratio", type=float, default=0.95)
    parser.add_argument("--max_norm", type=float, default=1.0,
                        help="learning rate of adam")
    parser.add_argument("--clip_grad_norm", type=float, default=None,
                        help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0,
                        help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="adam first beta value")

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    train(args)
