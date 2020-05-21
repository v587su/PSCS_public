import torch
import torch.nn as nn
import torch.nn.functional as F


class RankLoss(nn.Module):
    def __init__(self, config):
        super(RankLoss, self).__init__()
        self.margin = config.margin

    def forward(self, nl_vec_pos, nl_vec_neg, code_vec_pos, code_vec_neg):
        return (self.margin - F.cosine_similarity(nl_vec_pos,
                                                  code_vec_pos) + F.cosine_similarity(
            nl_vec_neg, code_vec_pos)).clamp(min=1e-6).mean()

