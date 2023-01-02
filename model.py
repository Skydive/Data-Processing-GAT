import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sparse
from GCNLayer import GCNLayer
from GATLayer import GATLayer
class GCNModel(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes, dropout, use_bias=True):
        super(GCNModel, self).__init__()
        self.gcn_1 = GCNLayer(node_features, hidden_dim, use_bias)
        self.gcn_2 = GCNLayer(hidden_dim, num_classes, use_bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj):
        x = self.gcn_1(x, adj)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gcn_2(x, adj)
        return x
        
class GATModel(nn.Module):
    def __init__(self, node_features, hidden_dim, num_classes, dropout, alpha, num_heads):
        super(GATModel, self).__init__()
        self.dropout = dropout

        self.attentions = [GATLayer(node_features, hidden_dim, dropout=dropout, alpha=alpha, concat=True) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GATLayer(hidden_dim * num_heads, num_classes, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)