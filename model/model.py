import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init

from torch import einsum


class PSCSNetwork(nn.Module):
    def __init__(self, config, nl_vocab_len, path_vocab_len, device):
        super(PSCSNetwork, self).__init__()
        self.config = config
        self.nl_emb = nn.Embedding(nl_vocab_len, config.emb_size,
                                   padding_idx=0, max_norm=config.max_norm)
        self.path_emb = nn.Embedding(path_vocab_len, config.emb_size,
                                     padding_idx=0, max_norm=config.max_norm)
        self.linear = nn.Linear(config.emb_size * 4, config.emb_size)
        self.path_lstm = nn.LSTM(config.emb_size, config.emb_size,
                                 num_layers=config.num_layers,
                                 dropout=config.rnn_dropout,
                                 batch_first=True, bidirectional=True)
        self.W_a = nn.Parameter(
            torch.rand((config.emb_size * 4, config.emb_size * 4),
                       dtype=torch.float,
                       device=device, requires_grad=True))
        self.W_b = nn.Parameter(torch.rand((config.emb_size, config.emb_size),
                                           dtype=torch.float,
                                           device=device, requires_grad=True))
        self.dropout = nn.Dropout(config.dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.W_a)
        nn.init.xavier_uniform_(self.W_b)
        nn.init.uniform_(self.nl_emb.weight, -0.1, 0.1)
        nn.init.constant_(self.nl_emb.weight[0], 0)
        nn.init.uniform_(self.path_emb.weight, -0.1, 0.1)
        nn.init.constant_(self.path_emb.weight[0], 0)
        for w in self.path_lstm.parameters():  # initialize the gate weights with orthogonal
            if w.dim() > 1:
                weight_init.orthogonal_(w)

    def forward(self, nl, path, start_token, end_token, epoch=0):

        batch_size = nl.size(0)

        nl_emb = self.nl_emb(nl)

        start_token_emb = self.nl_emb(
            start_token.view(-1, self.config.nl_seq_len),
        )
        start_token_emb = start_token_emb.sum(1)

        end_token_emb = self.nl_emb(
            end_token.view(-1, self.config.nl_seq_len),
        )
        end_token_emb = end_token_emb.sum(1)

        path_emb = self.path_emb(
            path.view(-1, self.config.path_len))
        _, (hidden, c_n) = self.path_lstm(path_emb)
        hidden = hidden[-2:, :, :]

        hidden = hidden.permute((1, 0, 2)).contiguous().view(
            batch_size * self.config.k, 1, -1)
        path_vec = hidden.squeeze(1)
        code_emb = torch.cat((start_token_emb, end_token_emb, path_vec), 1)
        code_emb = self.dropout(code_emb)

        nl_vec = self.get_nl_vec(nl_emb)
        code_vec = self.get_code_vec(code_emb)
        code_vec = self.linear(code_vec)
        return nl_vec, code_vec

    def attention(self, encoder_output_bag, hidden, lengths_k, w):

        """
        encoder_output_bag : (batch, k, hidden_size) bag of embedded ast path
        hidden : (1 , batch, hidden_size):
        lengths_k : (batch, 1) length of k in each example
        """

        # e_out : (batch * k, hidden_size)

        e_out = torch.cat(encoder_output_bag, dim=0)

        # e_out : (batch * k(i), hidden_size(j))
        # self.W_a  : [hidden_size(j), hidden_size(k)]
        # ha -> : [batch * k(i), hidden_size(k)]
        ha = einsum('ij,jk->ik', e_out, w)

        # ha -> : [batch, (k, hidden_size)]
        ha = torch.split(ha, lengths_k, dim=0)

        # _ha : (k(i), hidden_size(j))
        # dh = [batch, (1, hidden_size)]
        hd = hidden.transpose(0, 1)
        hd = torch.unbind(hd, dim=0)
        # _hd : (1(k), hidden_size(j))
        # at : [batch, ( k(i) ) ]
        at = [F.softmax(torch.einsum('ij,kj->i', _ha, _hd), dim=0) for
              _ha, _hd in zip(ha, hd)]

        # a : ( k(i) )
        # e : ( k(i), hidden_size(j))
        # ct : [batch, (hidden_size(j)) ] -> [batch, (1, hidden_size) ]
        ct = [torch.einsum('i,ij->j', a, e).unsqueeze(0) for a, e in
              zip(at, encoder_output_bag)]

        # ct [batch, hidden_size(k)]
        # -> (1, batch, hidden_size)
        ct = torch.cat(ct, dim=0).unsqueeze(0)

        return ct

    def get_nl_vec(self, nl_emb):
        nl_emb = torch.split(nl_emb.view(-1, self.config.emb_size),
                             self.config.nl_seq_len, dim=0)
        hidden_0 = [ne.mean(0).unsqueeze(dim=0) for ne in nl_emb]
        hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)
        nl_emb_atten = self.attention(nl_emb, hidden_0,
                                      self.config.nl_seq_len, self.W_b)
        nl_emb = nl_emb_atten[-1]
        return nl_emb

    def get_code_vec(self, code_emb):
        code_vec = torch.split(
            code_emb.contiguous().view(-1, 4 * self.config.emb_size),
            self.config.k, dim=0)
        hidden_0 = [cv.mean(0).unsqueeze(dim=0) for cv in code_vec]
        hidden_0 = torch.cat(hidden_0, dim=0).unsqueeze(dim=0)
        code_vec_atten = self.attention(code_vec, hidden_0, self.config.k,
                                        self.W_a)
        return code_vec_atten[-1]
