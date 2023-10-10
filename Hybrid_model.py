import pymongo
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import community
import sys

from Baseline1.recommender import user_studies
from Baseline1.content_based import ContentBasedModule
from Baseline1.crawler import Crawler
from Baseline1.collaborative_filtering import CollaborativeFilteringModule
from Baseline1.recommender import Recommender
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torch_geometric.datasets import KarateClub
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch.optim as optim


# Define neural network architecture for the hybrid model
class GCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim):
        super().__init__()

        # Define the layers of  neural network here
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        h = h.tanh()
        h = self.conv2(h, edge_index)
        h = h.tanh()  # Final GNN embedding space.

        # Apply a final (linear) classifier.
        out = self.classifier(h)

        return out, h

def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()

if __name__ == '__main__':
    import pprint
    # dataset = KarateClub()
    # data = dataset[0]
    # pprint.pprint(data.x)
    # pprint.pprint(data.edge_index)
    # pprint.pprint(data.y)
    # pprint.pprint(data.train_mask)
    # pprint.pprint(data.y[data.train_mask])
    # a = 1 / 0

    # Connect to MongoDB for the citation network data
    print("connecting to mongo")
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client["Aminer"]
    collection = db["papers"]
    data = list(collection.find())
    papers_dict = {paper["id"]: paper for paper in data}

    # Define paper of interest
    crawler = Crawler(papers_dict)
    content_based = ContentBasedModule()
    paper_id = "556798"
    paper = papers_dict[paper_id]
    paper_tf = content_based.term_freq(paper)

    # finding subset to build graph out of
    subset, candidates = crawler.get_subset(paper_id)
    collab_filter = CollaborativeFilteringModule(subset)

    print("building citation network")
    new_index = {}
    features = []
    edge_index = [[], []]

    for id in subset:
        paper = subset[id]
        if id not in new_index:
            new_index[id] = len(features)
            cosine = content_based.cosine_simi(paper_tf, content_based.term_freq(id))
            cooccured = collab_filter.get_cooccurred_score(paper_id, id)
            cooccuring = collab_filter.get_cooccurring_score(paper_id, id)
            features.append([cosine, cooccured, cooccuring])
        for ref in paper.get('references', []):
            if ref in subset:
                if ref not in new_index:
                    new_index[ref] = len(features)
                    cosine = content_based.cosine_simi(paper_tf, content_based.term_freq(ref))
                    cooccured = collab_filter.get_cooccurred_score(paper_id, ref)
                    cooccuring = collab_filter.get_cooccurring_score(paper_id, ref)
                    features.append([cosine, cooccured, cooccuring])
                edge_index[0].append(new_index[id])
                edge_index[1].append(new_index[ref])

    # Retrieve temp ideal papers
    recommendation_system = Recommender(papers_dict)

    # Get recommendations for the target paper (including similarity scores)
    recommendations = recommendation_system.recommend(paper_id)
    y = [0 for i in range(len(features))]
    for paper in recommendations:
        y[new_index[paper["id"]]] = 1
    x = torch.tensor(features, dtype=torch.float)

    train_mask = [True if i < 20 else False for i in range(len(features))]
    rec_ids = set()
    for recommendation in recommendations:
        train_mask[new_index[recommendation["id"]]] = True
        rec_ids.add(recommendation["id"])

    # Calculate the number of unique years
    num_classes = 2
    num_features = 3
    hidden_dim = 10
    model = GCN(num_features, num_classes, hidden_dim)

    # Define loss functions and optimizer
    criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

    # Prepare data and create DataLoader
    data = Data(x=x,
                edge_index=torch.tensor(edge_index),
                y=torch.tensor(y, dtype=torch.long))

    # Training loop (customize as needed)
    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear gradients.
        out, h = model(data.x, data.edge_index)  # Perform a single forward pass.
        loss = criterion(out[train_mask], data.y[train_mask])  # Compute the loss solely based on the training nodes.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.

        print(f'Epoch [{epoch + 1}/{epochs}] Loss: {loss.item()}')

    reverse_index = {new_index[i]: i for i in new_index}
    out, h = model(data.x, data.edge_index)
    pprint.pprint(out)
    j = 0
    test_recs = set()
    for i in out:
        if i[0] < i[1]:
            test_recs.add(reverse_index[j])
        j += 1

    print(f"length of optimal: {len(rec_ids)}")
    print(f"Length of recommendations: {len(test_recs)}")
    intersect = test_recs.intersection(rec_ids)
    print("Intersection = ")
    pprint.pprint(intersect)

    visualize_embedding(h, color=data.y)
