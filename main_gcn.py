from model import GCNModel
from util import *
from globals import config
from visualization import *

from training import training_loop, evaluate_on_test

from tqdm import tqdm

if __name__ == "__main__":
    np.random.seed(1337)
    torch.random.manual_seed(1337)

    features, labels, adj, edges = load_data(config)
    adj = convert_scipy_to_torch_sparse(adj)
    # adj = torch.FloatTensor(np.array(adj.todense())) (Dense GAT)

    # visualize_graph(edges, labels.cpu().tolist(), save=False)
    NUM_CLASSES = int(labels.max().item() + 1)

    train_set_ind, val_set_ind, test_set_ind = prepare_dataset(labels, NUM_CLASSES, config)

    model = GCNModel(features.shape[1], config.hidden_dim, NUM_CLASSES, config.dropout, config.use_bias)

    tqdm.write(f"Started training with 1 run.");
    val_acc, val_loss = training_loop(model, features, labels, adj, train_set_ind, val_set_ind, config)
    out_features = evaluate_on_test(model, features, labels, adj, test_set_ind, config)

    visualize_validation_performance(val_acc, val_loss)
    print("TSNE");
    visualize_embedding_tSNE(labels, out_features, NUM_CLASSES)