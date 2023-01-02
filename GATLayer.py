import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sparse

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GATLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)));
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)));
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, adj):
        # Multiply by weights
        x = x @ self.W; # x is of form (N, in_features)
        # Output features (multiply by normalized adjacency matrix)
        return torch.sparse.mm(adj, x);