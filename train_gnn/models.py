from torch_geometric.nn import MessagePassing
import torch.nn as nn
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MPNN(MessagePassing):
    def __init__(self, dim_in, dim_out, aggr='max'):
        super().__init__(aggr=aggr)
        self.lin_tgt = nn.Linear(dim_in, dim_out)
        self.lin_src = nn.Linear(dim_in, dim_out)
        self.emb_self_edge = nn.Embedding(2, dim_out)
        self.relu = nn.ReLU()

    def forward(self, x_src, x_dst, es, weights=None):
        return self.relu(self.propagate(es, x=(x_src, x_dst),
                              size=(x_src.size(0),x_dst.size(0)),
                              weights=weights))

    def message(self, x_i, x_j, edge_index, weights=None):
        if weights is None:
            return self.lin_tgt(x_i) + self.lin_src(x_j) + self.emb_self_edge((edge_index[1] == edge_index[0]).long())
        else:
            output = (self.lin_tgt(x_i) + self.lin_src(x_j) + self.emb_self_edge((edge_index[1] == edge_index[0]).long()))
            output_weighted = weights * output
            return output_weighted


class MPNNModel(nn.Module):
    def __init__(self, dim_h, netypes, t_srcs, t_tgts,
                 n_al, dim_ac, n_layers=5, n_classes=3, device=device):
        super().__init__()
        self.enc_al = nn.Embedding(n_al, dim_h)
        self.enc_ac = nn.Linear(dim_ac, dim_h)

        nn.init.xavier_normal_(self.enc_al.weight)
        nn.init.xavier_normal_(self.enc_ac.weight)
        nn.init.normal_(self.enc_ac.bias)

        self.emb_test = nn.Embedding(1, dim_h)
        self.t_srcs = t_srcs
        self.t_tgts = t_tgts

        self.netypes = netypes
        self.n_layers = n_layers
        self.relu = nn.ReLU()
        self.mpnns = nn.ModuleList(
            [nn.ModuleList([MPNN(dim_h, dim_h) for _ in range(self.netypes)])
             for _ in range(self.n_layers)])
        self.decode = nn.Linear(dim_h, n_classes)
        if n_classes > 1:
            self.last_act = nn.Softmax(dim=1)
        else:
            self.last_act = nn.Sigmoid()

    def forward(self, xs, ess, weights=None):
        '''xs: word embeddings of the nodes'''
        for i in range(self.n_layers):
            out = [0, 0]
            for j, (es, t_src, t_tgt) in enumerate(
                    zip(ess, self.t_srcs, self.t_tgts)):
                out[t_tgt] += self.mpnns[i][j](xs[t_src], xs[t_tgt], es,
                                               weights[j] if weights else None)
            xs = self.relu(out)
        last = self.decode(xs)
        return last, self.last_act(last)

