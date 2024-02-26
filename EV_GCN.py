import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch_geometric as tg
import torch.nn.functional as F
from torch import nn 
from PAE import PAE
# from layer import GCNdenseConv
from SSG_CONV import SSGConv

class EV_GCN(torch.nn.Module):
    def __init__(self, input_dim, num_classes, dropout, edgenet_input_dim, edge_dropout, hgc, lg):
        super(EV_GCN, self).__init__()
        K = 3
        hidden = [hgc for i in range(lg)] 
        self.dropout = dropout
        self.edge_dropout = edge_dropout 
        bias = False
        self.relu = torch.nn.ReLU(inplace=True)
        self.lg = lg
        self.gconv = nn.ModuleList()
        for i in range(lg):
            in_channels = input_dim if i==0  else hidden[i-1]
            # self.gconv.append(tg.nn.ChebConv(in_channels, hidden[i], K, normalization='sym', bias=bias))

            # self.gconv.append(tg.nn.GCNConv(in_channels, hidden[i], bias=bias))
            # self.gconv.append(tg.nn.GraphConv(in_channels, hidden[i],aggr='mean', bias=True))
            self.gconv.append(tg.nn.GATConv(in_channels, hidden[i],heads=6,dropout=0.2,concat=False,negative_slope=0.3, bias=bias))
            # self.gconv.append(tg.nn.TAGConv(in_channels, hidden[i], K=9, bias=bias))
            # self.gconv.append(tg.nn.ARMAConv(in_channels, hidden[i],num_stacks=3,num_layers=5, bias=bias))
            # self.gconv.append(SSGConv(in_channels, hidden[i], 0.3, bias=bias))

            # self.gconv.append(tg.nn.SGConv(in_channels, hidden[i], alpha=0.3, K=5, bias=bias))
            # self.gconv.append(tg.nn.ClusterGCNConv(in_channels, hidden[i], diag_lambda = 3.5))
            # self.gconv.append(tg.nn.EGConv(in_channels, hidden[i],num_heads=4))

            # self.gconv.append(SSGConv(in_channels, hidden[i],alpha=0.3,K=2, bias=bias))
            # self.gconv.append(tg.nn.GatedGraphConv(hidden[i],num_layers=1, bias=bias))
            # self.gconv.append(tg.nn.APPNP(K=K, alpha=0.1))
            # self.gconv.append(GCNdenseConv(in_channels, hidden[i], bias=bias))
            # self.gconv.append(tg.nn.GIN(in_channels, hidden[i], 3))
            # self.gconv.append(tg.nn.EGConv(in_channels, hidden[i]))
        cls_input_dim = sum(hidden)

        self.cls = nn.Sequential(
                # torch.nn.Linear(cls_input_dim, 256),
                torch.nn.Linear(hgc, 256),
                torch.nn.ReLU(inplace=True),
                nn.BatchNorm1d(256), 
                torch.nn.Linear(256, num_classes))

        self.edge_net = PAE(input_dim=edgenet_input_dim//2, dropout=dropout)
        self.model_init()

        # add
        self.attention = Attention(hgc)

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, features, edge_index, edgenet_input, enforce_edropout=False): 
        if self.edge_dropout>0:
            if enforce_edropout or self.training:
                one_mask = torch.ones([edgenet_input.shape[0],1]).cuda()
                self.drop_mask = F.dropout(one_mask, self.edge_dropout, True)
                self.bool_mask = torch.squeeze(self.drop_mask.type(torch.bool))
                edge_index = edge_index[:, self.bool_mask]
                edgenet_input = edgenet_input[self.bool_mask] 
            
        edge_weight = torch.squeeze(self.edge_net(edgenet_input))
        features = F.dropout(features, self.dropout, self.training)
        h = self.relu(self.gconv[0](features, edge_index, edge_weight))
        # h = self.relu(self.gconv[0](features, edge_index))
        h0 = h

        list = []
        list.append(h0)
        # for i in range(1, self.lg):
        #     h = F.dropout(h, self.dropout, self.training)
        #     # h= self.relu(self.gconv[i](h, edge_index, edge_weight))
        #     h = self.relu(self.gconv[i](h, edge_index))
        #     jk = torch.cat((h0, h), axis=1)
        #     # jk = torch.stack((h0, h), axis=1) # 871 lg 16
        #     h0 = jk
        for i in range(1, self.lg):
            h = F.dropout(h, self.dropout, self.training)
            h = self.relu(self.gconv[i](h, edge_index, edge_weight))
            # h = self.relu(self.gconv[i](h, edge_index))
            list.append(h)
            # jk = torch.cat((h0, h), axis=1)
            # jk = torch.stack((h0, h), axis=1) # 871 lg hgc
            # h0 = jk

        emb = torch.stack(list, dim=1)
        emb, att = self.attention(emb)

        # add
        # _,a = self.attention(jk)
        # jk = jk*a

        logit = self.cls(emb)
        # logit = self.cls(h)
        # logit = self.cls(jk)

        return logit, edge_weight

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)  # 871xlgx256
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta