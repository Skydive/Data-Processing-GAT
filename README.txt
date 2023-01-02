Uses a nix flake for total atomic reproducibility.

Alternatively one may run:
>pip install -r requirements.txt

To run Graph Convolutional Network:
>python main_gcn.py

To run Graph Attention Network:
>python main_gat.py

Results: (200 Epochs)
GCN: (0:05 runtime) [Since sparse matrix implementation]
Loss: 0.655
Accuracy: 0.80

GAT: (2:52 runtime) [Since dense matrix implementation]
Loss: 0.647 
Accuracy: 0.81