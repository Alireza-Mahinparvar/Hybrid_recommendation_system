import pymongo
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import pickle

print("loading pickle")
papers_dict = pickle.load(open("papers_dict.p", "rb"))

print("loading references")
reverse_references = {}
for paper in papers_dict:
    for ref_id in papers_dict[paper].get("references", []):
        if ref_id not in reverse_references:
            reverse_references[ref_id] = []
        reverse_references[ref_id].append(paper)


# Helper functions
def find_paper_by_id(paper_id):
    return papers_dict.get(paper_id)


def find_citing_papers(paper_id):
    citing_paper_ids = reverse_references.get(paper_id, [])
    return [find_paper_by_id(paper_id) for paper_id in citing_paper_ids]


# Main function to build citation network
def build_citation_network(paper, forward_levels=5, backward_levels=5):
    citation_network = nx.DiGraph()
    citation_network.add_node(paper["id"], **paper)

    # Forward direction (cited papers)
    queue = deque([(paper, 0)])
    while queue:
        current_paper, level = queue.popleft()

        if level < forward_levels:
            for cited_paper_id in current_paper.get("references", []):
                cited_paper = find_paper_by_id(cited_paper_id)
                if cited_paper and not citation_network.has_node(cited_paper_id):
                    citation_network.add_node(cited_paper_id, **cited_paper)
                    queue.append((cited_paper, level + 1))
                citation_network.add_edge(current_paper["id"], cited_paper_id)

    # Backward direction (citing papers)
    queue = deque([(paper, 0)])
    while queue:
        current_paper, level = queue.popleft()

        if level < backward_levels:
            citing_papers = find_citing_papers(current_paper["id"])
            for citing_paper in citing_papers:
                if citing_paper and not citation_network.has_node(citing_paper["id"]):
                    citation_network.add_node(citing_paper["id"], **citing_paper)
                    queue.append((citing_paper, level + 1))
                citation_network.add_edge(citing_paper["id"], current_paper["id"])

    return citation_network


# Custom layout function for visualizing the citation network
def bidirectional_layered_layout(G, root):
    layers = {root: 0}
    queue = deque([root])

    # Forward layers
    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in layers:
                layers[neighbor] = layers[node] + 1
                queue.append(neighbor)

    # Backward layers
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for predecessor in G.predecessors(node):
            if predecessor not in layers:
                layers[predecessor] = layers[node] - 1
                queue.append(predecessor)

    pos = {}
    for node, layer in layers.items():
        pos[node] = (layer, hash(node) % 1000)  # Using the hash to distribute nodes vertically

    return pos


# Functions for bibliographic coupling and co - citation analysis
def bibliographic_coupling(G, X, Y):
    X_references = set(G.neighbors(X))
    Y_references = set(G.neighbors(Y))
    return len(X_references.intersection(Y_references))


def co_citation(G, X, Y):
    X_cited_by = set(G.predecessors(X))
    Y_cited_by = set(G.predecessors(Y))
    return len(X_cited_by.intersection(Y_cited_by))


# Functions for calculating candidate score

def candidate_score(G, paper_of_interest, paper):
    if nx.has_path(citation_network, source=paper_of_interest, target=paper):
        distance = nx.shortest_path_length(citation_network, source=paper_of_interest, target=paper)
    else:
        distance = float('inf')  # or some other large number indicating a very long distance
    # distance = nx.shortest_path_length(G, source=paper_of_interest, target=paper)
    bc = bibliographic_coupling(G, paper_of_interest, paper)
    cc = co_citation(G, paper_of_interest, paper)
    return (bc + cc) / distance


# Example usage
paper = papers_dict[1872]  # Replace 110 with the id of the paper you want to start with
print("building citation network")
citation_network = build_citation_network(paper)
paper_of_interest = paper["id"]
# print(citation_network.nodes)
# print(citation_network.edges)

print("paper of interest")
print(paper)
candidate_papers = []

# Draw and display the graph
pos = bidirectional_layered_layout(citation_network, paper["id"])
# plt.figure(figsize=(12, 12))
# nx.draw(citation_network, pos, node_size=50, font_size=8, with_labels=True, node_color="skyblue", edge_color="gray",
#         arrows=True)
# plt.title("Citation Network")
# plt.savefig("citation_network.png", dpi=300, bbox_inches="tight")
# plt.show()

for other_paper in citation_network.nodes:
    if other_paper != paper_of_interest:
        bc = bibliographic_coupling(citation_network, paper_of_interest, other_paper)
        cc = co_citation(citation_network, paper_of_interest, other_paper)
        score = candidate_score(citation_network, paper_of_interest, other_paper)
        if nx.has_path(citation_network, source=paper_of_interest, target=other_paper):
            distance = nx.shortest_path_length(citation_network, source=paper_of_interest, target=other_paper)
        else:
            distance = float('inf')  # or some other large number indicating a very long distance
        # distance = nx.shortest_path_length(citation_network, source=paper_of_interest, target=other_paper)

        candidate_papers.append({
            "paper": other_paper,
            "bibliographic_coupling": bc,
            "co_citation": cc,
            "score": score,
            "distance": distance,
        })


# print(candidate_papers)


def rank_papers(candidate_papers, top_n=10):
    G = nx.DiGraph()
    G.add_edges_from((paper["paper"], paper_of_interest) for paper in candidate_papers)

    degree_centrality = nx.in_degree_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

    centrality_data = []

    for paper in candidate_papers:
        paper_id = paper["paper"]
        degree = degree_centrality[paper_id]
        closeness = closeness_centrality[paper_id]
        betweenness = betweenness_centrality[paper_id]
        eigenvector = eigenvector_centrality[paper_id]

        centrality_data.append({
            "paper": paper_id,
            "degree": degree,
            "closeness": closeness,
            "betweenness": betweenness,
            "eigenvector": eigenvector
        })

    centrality_df = pd.DataFrame(centrality_data)
    centrality_df["degree_rank"] = centrality_df["degree"].rank(ascending=False)
    centrality_df["closeness_rank"] = centrality_df["closeness"].rank(ascending=False)
    centrality_df["betweenness_rank"] = centrality_df["betweenness"].rank(ascending=False)
    centrality_df["eigenvector_rank"] = centrality_df["eigenvector"].rank(ascending=False)

    centrality_df["average_rank"] = centrality_df[
        ["degree_rank", "closeness_rank", "betweenness_rank", "eigenvector_rank"]].mean(axis=1)
    centrality_df = centrality_df.sort_values("average_rank")

    top_papers = centrality_df[["paper", "average_rank"]].head(top_n).to_dict("records")

    return top_papers


# View top  10
print("Top 10")
top_papers = rank_papers(candidate_papers)
# print("Top papers")

for rank, paper in enumerate(top_papers, start=1):
    id = paper['paper']
    paper_info = papers_dict.get(id, None)
    if paper_info:
        title = paper_info.get('title', "Title not found")
    else:
        title = "Paper information not found"
    print(f"Rank {rank} - Paper ID: {id}, Title: {title}")

author_dict = pickle.load(open("author_dict.p", "rb"))
ground_truth = pickle.load(open("ground_truth.p", "rb"))

print("generating test data")
test_data = []
for author in ground_truth:
    print(f"adding author {author} to test data")
    test_data = test_data + list(set(author_dict[author]) - set(test_data))

# Initialize lists to store evaluation results
precision_scores = []
recall_scores = []
map_scores = []

print("Beginning test iterations")
for x in range(1000):
    print(f"on paper {x}")
    poid = test_data[x]
    if poid not in papers_dict:
        continue
    poi = papers_dict[poid]

    actual = []
    for author in poi.get("author", []):
        ex = ground_truth.get(author, [])
        actual = actual + list(set(ex) - set(actual))

    # Calculate precision, recall, and MAP for each test paper individually
    if len(actual) > 0:
        paper = papers_dict[1872]  # Replace 110 with the id of the paper you want to start with
        print("building citation network")
        citation_network = build_citation_network(paper)
        paper_of_interest = paper["id"]
        candidate_papers = []
        for other_paper in citation_network.nodes:
            if other_paper != paper_of_interest:
                bc = bibliographic_coupling(citation_network, paper_of_interest, other_paper)
                cc = co_citation(citation_network, paper_of_interest, other_paper)
                score = candidate_score(citation_network, paper_of_interest, other_paper)
                if nx.has_path(citation_network, source=paper_of_interest, target=other_paper):
                    distance = nx.shortest_path_length(citation_network, source=paper_of_interest, target=other_paper)
                else:
                    distance = float('inf')  # or some other large number indicating a very long distance
                # distance = nx.shortest_path_length(citation_network, source=paper_of_interest, target=other_paper)
                print(f"adding candidate {other_paper}")
                candidate_papers.append({
                    "paper": other_paper,
                    "bibliographic_coupling": bc,
                    "co_citation": cc,
                    "score": score,
                    "distance": distance,
                })
        print(actual)
        print("ranking")
        top_papers = rank_papers(candidate_papers)

        rec_ids = [paper["paper"] for paper in top_papers]
        print(rec_ids)

        true_positives = len(set(actual).intersection(rec_ids))
        precision = true_positives / len(top_papers) if len(top_papers) > 0 else 0
        recall = true_positives / len(actual) if len(actual) > 0 else 0
        map_score = 1 if true_positives > 0 else 0  # Set MAP to 1 if there is at least one true positive

        precision_scores.append(precision)
        recall_scores.append(recall)
        map_scores.append(map_score)

# Calculate and print average evaluation metrics
avg_precision = sum(precision_scores) / len(precision_scores)
avg_recall = sum(recall_scores) / len(recall_scores)
avg_map = sum(map_scores) / len(map_scores)

print(f"Average Precision: {avg_precision:.4f}")
print(f"Average Recall: {avg_recall:.4f}")
print(f"Average MAP: {avg_map:.4f}")