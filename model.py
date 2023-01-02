import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sparse
from GCNLayer import GCNLayer

class GCNModel(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes, dropout, alpha, use_bias=True):
        super(GCNModel, self).__init__()
        self.alpha = alpha
        self.gcn_1 = GCNLayer(node_features, hidden_dim, use_bias)
        self.gcn_2 = GCNLayer(hidden_dim, num_classes, use_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = self.gcn_1(x, adj)
        x = F.leaky_relu(x, self.alpha)
        x = self.dropout(x)
        x = self.gcn_2(x, adj)
        return x
        