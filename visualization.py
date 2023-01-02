import numpy as np
import igraph as ig
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Color map for each class
cora_label_to_color_map = {0: "red", 1: "blue", 2: "green",
                           3: "orange", 4: "yellow", 5: "pink", 6: "gray"}


def visualize_embedding_tSNE(labels, out_features, num_classes):
    node_labels = labels.cpu().numpy()
    out_features = out_features.cpu().numpy()
    t_sne_embeddings = TSNE(n_components=2, perplexity=30, method='barnes_hut').fit_transform(out_features)

    plt.figure()
    for class_id in range(num_classes):
        plt.scatter(t_sne_embeddings[node_labels == class_id, 0],
                    t_sne_embeddings[node_labels == class_id, 1], s=20,
                    color=cora_label_to_color_map[class_id],
                    edgecolors='black', linewidths=0.15)

    plt.axis("off")
    plt.title("t-SNE projection of the learned features")
    plt.show()

def visualize_validation_performance(val_acc, val_loss):
    f, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    axs[0].plot(val_loss, linewidth=2, color="red")
    axs[0].set_title("Validation loss")
    axs[0].set_ylabel("Cross Entropy Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].grid()

    axs[1].plot(val_acc, linewidth=2, color="red")
    axs[1].set_title("Validation accuracy")
    axs[1].set_ylabel("Acc")
    axs[1].set_xlabel("Epoch")
    axs[1].grid()

    plt.show()