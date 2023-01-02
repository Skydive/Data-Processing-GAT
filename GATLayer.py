import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sparse

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GATLayer, self).__init__()
        self.concat = concat
        self.out_features = out_features
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = alpha
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

        self.leaky_relu = nn.LeakyReLU(self.alpha)


    def forward(self, x, adj):
        # Multiply by weights
        Wh = x @ self.W; # x is of form (N, in_features)

        # Calculate attention coefficients
        Wh1 = Wh @ self.a[:self.out_features, :]
        Wh2 = Wh @ self.a[self.out_features:, :]
        e = Wh1 + Wh2.T
        e = self.leaky_relu(e)

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        out = attention @ Wh

        if self.concat:
            return F.elu(out)
        else:
            return out

