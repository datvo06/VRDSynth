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
    def __init__(self, dim_h, netypes, 
                 dim_in, n_layers=5, n_classes=7, device=device):
        super().__init__()
        self.enc = nn.Embedding(dim_in, dim_h)
        nn.init.xavier_normal_(self.enc.weight)
        nn.init.normal_(self.enc.bias)

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
        self.h = dim_h

    def forward(self, x, es, weights=None):
        '''xs: word embeddings of the nodes'''
        x = self.enc(x)
        for i in range(self.n_layers):
            out = torch.zeros(x.size(0), self.h).to(device)
            for j in range(self.netypes):
                out += self.mpnns[i][j](x, x, es[j], weights)
            x = self.relu(out)
        last = self.decode(x)
        return last, self.last_act(last)
