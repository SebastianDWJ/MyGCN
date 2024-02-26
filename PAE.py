import torch
from torch.nn import Linear as Lin, Sequential as Seq
import torch.nn.functional as F
from torch import nn

class PAE(torch.nn.Module):
    def __init__(self, input_dim, dropout=0.2):
        super(PAE, self).__init__()
        hidden=128
        self.parser =nn.Sequential(
                nn.Linear(input_dim, hidden, bias=True),
                # nn.RELU(inplace=True),
                nn.GELU(), # gelu
                nn.BatchNorm1d(hidden),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden, bias=True),
                )
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        self.input_dim = input_dim
        self.model_init()
        self.relu = nn.ReLU(inplace=True)
        self.elu = nn.ReLU()

        # add
        # self.attention = Attention(hidden)
        self.attention = Attention(3)

    def forward(self, x):
        x1 = x[:,0:self.input_dim]
        x2 = x[:,self.input_dim:]
        # add
        _,a1 = self.attention(x1)
        _,a2 = self.attention(x2)
        x1 = a1*x1
        x2 = a2*x2


        h1 = self.parser(x1) 
        h2 = self.parser(x2)

        # add
        # _, a1 = self.attention(h1)
        # _, a2 = self.attention(h2)
        # h1 = a1*h1
        # h2 = a2*h2
        p = (self.cos(h1, h2) + 1)*0.5
        return p

    def model_init(self):
        for m in self.modules():
            if isinstance(m, Lin):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True



class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
