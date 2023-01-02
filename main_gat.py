import torch
import torch.nn as nn

import numpy as np
import time

from model import GATModel
from util import *
from globals import config
from visualization import *

from tqdm import tqdm

def training_loop(model, features, labels, adj, train_set_ind, val_set_ind, config):
    if config.cuda:
        model.cuda()
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr,
                                 weight_decay=config.weight_decay)
    criterion = nn.CrossEntropyLoss()

    validation_acc = []
    validation_loss = []

    t_start = time.time()
    for epoch in tqdm(range(config.epochs)):
        optimizer.zero_grad()
        model.train()

        y_pred = model(features, adj)
        train_loss = criterion(y_pred[train_set_ind], labels[train_set_ind])
        train_acc = accuracy(y_pred[train_set_ind], labels[train_set_ind])
        train_loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            val_loss = criterion(y_pred[val_set_ind], labels[val_set_ind])
            val_acc = accuracy(y_pred[val_set_ind], labels[val_set_ind])

            validation_loss.append(val_loss.item())
            validation_acc.append(val_acc)

        if not config.multiple_runs:
            tqdm.write(" | ".join([f"Epoch: {epoch:4d}", f"Train loss: {train_loss.item():.3f}",
                              f"Train acc: {train_acc:.2f}",
                              f"Val loss: {val_loss.item():.3f}",
                              f"Val acc: {val_acc:.2f}"]))

    t_end = time.time()

    if not config.multiple_runs:
        tqdm.write(f"Total training time: {t_end-t_start:.2f} seconds")

    return validation_acc, validation_loss


def evaluate_on_test(model, features, labels, adj, test_ind, config):
    if config.cuda:
        model.cuda()
        adj = adj.cuda()
        features = features.cuda()
        labels = labels.cuda()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        model.eval()
        y_pred = model(features, adj)
        test_loss = criterion(y_pred[test_ind], labels[test_ind])
        test_acc = accuracy(y_pred[test_ind], labels[test_ind])

    if not config.multiple_runs:
        tqdm.write("")
        tqdm.write(f"Test loss: {test_loss:.3f}  |  Test acc: {test_acc:.2f}")
        return y_pred
    else:
        return test_acc.item(), test_loss.item()



if __name__ == "__main__":
    np.random.seed(1337)
    torch.random.manual_seed(1337)

    features, labels, adj, edges = load_data(config)
    # adj = convert_scipy_to_torch_sparse(adj)
    adj = torch.FloatTensor(np.array(adj.todense()))

    # visualize_graph(edges, labels.cpu().tolist(), save=False)
    NUM_CLASSES = int(labels.max().item() + 1)

    train_set_ind, val_set_ind, test_set_ind = prepare_dataset(labels, NUM_CLASSES, config)

    model = GATModel(features.shape[1], config.hidden_dim, NUM_CLASSES, config.dropout, config.alpha, config.head_count)

    tqdm.write(f"Started training with 1 run.");
    val_acc, val_loss = training_loop(model, features, labels, adj, train_set_ind, val_set_ind, config)
    out_features = evaluate_on_test(model, features, labels, adj, test_set_ind, config)

    visualize_validation_performance(val_acc, val_loss)
    print("TSNE");
    visualize_embedding_tSNE(labels, out_features, NUM_CLASSES)