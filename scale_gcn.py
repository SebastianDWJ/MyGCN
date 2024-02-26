import math
import os
from tqdm import tqdm
import torch
from torch.nn import Linear
import torch.nn.functional as F
from layers import ScaleStarGConv, ScaleGConv
from models.mlp import MLP


class ScaleGCN(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, hidden_channels:int,
                 num_layers:int, dropedge:float=0.0, dropout:float=0.5, fix_beta:bool=False,
                 alpha:float=0.1, beta_base:float=0.5, cached=True, gconv_type="normal"):
        super(ScaleGCN, self).__init__()

        # arguments 
        self.in_channels        : int   = in_channels
        self.out_channels       : int   = out_channels
        self.hidden_channels    : int   = hidden_channels
        self.num_layers         : int   = num_layers
        self.dropout            : float = dropout
        self.alpha              : float = alpha
        self.beta_base          : float = beta_base
        self.fix_beta           : bool  = fix_beta
        assert gconv_type in ["normal", "star"]
        conv_op = ScaleGConv if gconv_type == "normal" else ScaleStarGConv

        self.init_layer = Linear(in_channels, hidden_channels)
        layers = []
        for _ in range(num_layers):
            layers.append(conv_op(hidden_channels, dropedge=dropedge, cached=cached))
        self.layers = torch.nn.ModuleList(layers)
        self.final_layers = Linear(hidden_channels, out_channels)

        self.reg_params = list(self.layers.parameters())
        self.non_reg_params = list(self.init_layer.parameters()) + list(self.final_layers.parameters())

    def forward(self, data, use_softmax=True):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.init_layer(x))
        init_x = x

        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.fix_beta:
                beta = self.beta_base
            else:
                beta = math.log(self.beta_base / (i + 1) + 1)
            x = F.relu(self.layers[i](x, edge_index, init_x, self.alpha, beta))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_layers(x)
        if use_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

    def saveto(self, path):
        torch.save(self.state_dict(), path)

    def loadfrom(self, path):
        if not os.path.exists(path):
            raise ValueError("Path not exists")
        self.load_state_dict(torch.load(path))
    
    def dump_intermediate_results(self, data):
        intermediate_results = {}
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.init_layer(x))
        init_x = x
        intermediate_results["init"] = init_x

        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.fix_beta:
                beta = self.beta_base
            else:
                beta = math.log(self.beta_base / (i + 1) + 1)
            x = F.relu(self.layers[i](x, edge_index, init_x, self.alpha, beta))
            intermediate_results["layer_{}".format(i)] = x

        return intermediate_results


class ScaleGCN_Large(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, hidden_channels:int, num_layers:int,
                 num_final_update:int, fix_beta:bool=False, dropedge:float=0.0, dropout:float=0.5,
                 alpha:float=0.1, beta_base:float=0.5, cached=False, gconv_type="normal"):
        super(ScaleGCN_Large, self).__init__()

        # arguments 
        self.in_channels        : int   = in_channels
        self.out_channels       : int   = out_channels
        self.hidden_channels    : int   = hidden_channels
        self.num_final_update   : int   = num_final_update
        self.num_layers         : int   = num_layers
        self.dropout            : float = dropout
        self.alpha              : float = alpha
        self.beta_base          : float = beta_base
        self.fix_beta           : bool  = fix_beta
        assert gconv_type in ["normal", "star"]
        conv_op = ScaleGConv if gconv_type == "normal" else ScaleStarGConv

        self.init_layer = Linear(in_channels, hidden_channels)
        conv_layers = []
        # ln_layers = []
        for _ in range(num_layers):
            conv_layers.append(conv_op(hidden_channels, dropedge=dropedge, cached=cached))
            # ln_layers.append(torch.nn.LayerNorm(hidden_channels, elementwise_affine=True))
        self.conv_layers = torch.nn.ModuleList(conv_layers)
        # self.ln_layers = torch.nn.ModuleList(ln_layers)
        self.final_layers = MLP(hidden_channels, out_channels, hidden_channels=hidden_channels,
                                num_layers=num_final_update, dropout=dropout, batchnorm=False, res_connect=True)

        self.reg_params = list(self.conv_layers.parameters()) + list(self.final_layers.parameters())
        self.non_reg_params = list(self.init_layer.parameters())

    def forward(self, data, use_softmax=True):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.init_layer(x))
        init_x = x

        for i in range(self.num_layers):
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.fix_beta:
                beta = self.beta_base
            else:
                beta = math.log(self.beta_base / (i + 1) + 1)
            x = self.conv_layers[i](x, edge_index, init_x, self.alpha, beta) + x
            # x = self.ln_layers[i](x)
            x = F.leaky_relu(x, negative_slope=0.1)
        x = self.final_layers(x)
        if use_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

    def saveto(self, path):
        torch.save(self.state_dict(), path)

    def loadfrom(self, path):
        if not os.path.exists(path):
            raise ValueError("Path not exists")
        self.load_state_dict(torch.load(path))


from layers import BiScaleStarGConv
class BiScaleGCN(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, hidden_channels:int,
                 num_layers:int, num_final_update:int, dropout:float=0.5):
        super(BiScaleGCN, self).__init__()

        self.in_channels        : int   = in_channels
        self.out_channels       : int   = out_channels
        self.hidden_channels    : int   = hidden_channels
        self.num_final_update   : int   = num_final_update
        self.num_layers         : int   = num_layers
        self.dropout            : float = dropout

        self.init_layer = Linear(in_channels, hidden_channels)
        layers = []
        for _ in range(num_layers):
            layers.append(BiScaleStarGConv(hidden_channels, dropout=dropout))
        self.convs = torch.nn.ModuleList(layers)
        self.final_layers = MLP(hidden_channels, out_channels, hidden_channels,
                                num_layers=num_final_update, dropout=dropout,
                                batchnorm=False, res_connect=True)
        
    def forward(self, data, use_softmax=True):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.init_layer(x))

        for i in range(self.num_layers):
            # x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            x = F.leaky_relu(x, negative_slope=0.1)
        x = self.final_layers(x)
        if use_softmax:
            return F.log_softmax(x, dim=1)
        else:
            return x

    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=x_all.size(0) * self.num_layers)

        first_func = lambda x: F.relu(self.init_layer(F.dropout(x, p=self.dropout, training=self.training)))
        x_all = self.batch_node_transform(x_all, subgraph_loader.batch_size, first_func, device)

        for i, layer in enumerate(self.convs):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = layer((x, x_target), edge_index, size=size)
                x = F.leaky_relu(x, negative_slope=0.1)
                xs.append(x.cpu())

                pbar.update(batch_size)
            x_all = torch.cat(xs, dim=0)
        
        x_all = self.batch_node_transform(x_all, subgraph_loader.batch_size, self.final_layers, device)
        pbar.close()
        return x_all

    def batch_node_transform(self, x, batch_size, func, device):
        total_num = x.size(0)
        num_batch = (total_num - 1) // batch_size + 1
        res = []

        for b in range(num_batch):
            s_idx = b * batch_size
            e_idx = min((b + 1) * batch_size, total_num)
            res.append(func(x[s_idx:e_idx].to(device)).cpu())
        x = torch.cat(res, dim=0)
        return x

    def saveto(self, path):
        torch.save(self.state_dict(), path)

    def loadfrom(self, path):
        if not os.path.exists(path):
            raise ValueError("Path not exists")
        self.load_state_dict(torch.load(path))
