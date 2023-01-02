from tqdm import tqdm
import torch
import numpy as np
import scipy.sparse as sparse

def accuracy(output, labels):
    y_pred = output.max(1)[1].type_as(labels)
    correct = y_pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def normalize_adj(adj):
    # A_tilde = A + I
    adj = adj + sparse.eye(adj.shape[-1]);

    D = np.array(adj.sum(1));
    D_invroot = np.power(D, -0.5).flatten();
    D_invroot[np.isinf(D_invroot)] = 0.0; # Remove INF
    D_invroot[np.isnan(D_invroot)] = 0.0; # Remove NaN
    D_invroot = sparse.diags(D_invroot, dtype=np.float32)
    return D_invroot @ adj @ D_invroot;

# Taken from: https://github.com/senadkurtisi/pytorch-GCN/blob/main/src/utils.py
def convert_scipy_to_torch_sparse(matrix):
    print("Matrix Type: ", type(matrix))
    matrix_helper_coo = matrix.tocoo().astype('float32')
    data = torch.FloatTensor(matrix_helper_coo.data)
    rows = torch.LongTensor(matrix_helper_coo.row)
    cols = torch.LongTensor(matrix_helper_coo.col)
    indices = torch.vstack([rows, cols])

    shape = torch.Size(matrix_helper_coo.shape)
    matrix = torch.sparse.FloatTensor(indices, data, shape)
    return matrix

def enumerate_labels(labels):
    unique = list(set(labels))
    labels = np.array([unique.index(label) for label in labels])
    return labels

def load_data(config):
    tqdm.write("Loading CORA data...")

    raw_nodes_data = np.genfromtxt(config.nodes_path, dtype="str")
    raw_node_ids = raw_nodes_data[:, 0].astype('int32')  # unique identifier of each node
    raw_node_labels = raw_nodes_data[:, -1]
    labels_enumerated = enumerate_labels(raw_node_labels)  # target labels as integers
    node_features = sparse.csr_matrix(raw_nodes_data[:, 1:-1], dtype="float32")
    
    ids_ordered = {raw_id: order for order, raw_id in enumerate(raw_node_ids)}
    raw_edges_data = np.genfromtxt(config.edges_path, dtype="int32")
    edges_ordered = np.array(list(map(ids_ordered.get, raw_edges_data.flatten())),
                             dtype='int32').reshape(raw_edges_data.shape)

    adj = sparse.coo_matrix((np.ones(edges_ordered.shape[0]), (edges_ordered[:, 0], edges_ordered[:, 1])),
                            shape=(labels_enumerated.shape[0], labels_enumerated.shape[0]),
                            dtype=np.float32)
    # Make the adjacency matrix symmetric
    adj = adj + adj.T.multiply(adj.T > adj)
    adj = normalize_adj(adj)

    features = torch.FloatTensor(node_features.toarray())
    labels = torch.LongTensor(labels_enumerated)
    
    tqdm.write("Dataset loaded successfully!")

    return features, labels, adj, edges_ordered

# TODO: Convert into data loader...
def prepare_dataset(labels, num_classes, config):
    # The original paper proposes that the training set is composed
    # out of 20 samples per class -> 140 samples, but the indices
    # above (range(140)) do not contain 20 samples per class
    # The remaining val/test indices were selected empirically
    classes = [ind for ind in range(num_classes)]
    train_set = []

    # Construct train set (indices) out of 20 samples per each class
    for class_label in classes:
        target_indices = torch.nonzero(labels == class_label, as_tuple=False).tolist()
        train_set += [ind[0] for ind in target_indices[:config.train_size_per_class]]

    # Extract the remaining samples
    validation_test_set = [ind for ind in range(len(labels)) if ind not in train_set]
    # Split the remaining samples into validation/test set
    validation_set = validation_test_set[:config.validation_size]
    test_set = validation_test_set[config.validation_size:config.validation_size+config.test_size]

    return train_set, validation_set, test_set
