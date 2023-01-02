import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sparse

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, use_bias=True):
        super(GCNLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W)
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(torch.zeros(size=(out_features,))))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        # Multiply by weights
        x = x @ self.W; # x is of form (N, in_features)
        if self.bias is not None:
            x += self.bias
        # Output features (multiply by normalized adjacency matrix)
        return torch.sparse.mm(adj, x)