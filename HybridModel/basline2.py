import networkx as nx
from collections import deque
import pandas as pd

class Baseline2:
    def __init__(self, data):
        self.papers_dict = {paper["id"]: paper for paper in data}
        self.reverse_references = {}
        for paper in data:
            for ref_id in paper.get("references", []):
                if ref_id not in self.reverse_references:
                    self.reverse_references[ref_id] = []
                self.reverse_references[ref_id].append(paper["id"])

    # Helper functions
    def find_paper_by_id(self, paper_id):
        return self.papers_dict.get(paper_id)


    def find_citing_papers(self, paper_id):
        citing_paper_ids = self.reverse_references.get(paper_id, [])
        return [self.find_paper_by_id(paper_id) for paper_id in citing_paper_ids]


    # Main function to build citation network
    def build_citation_network(self, paper, forward_levels=5, backward_levels=5):
        citation_network = nx.DiGraph()
        citation_network.add_node(paper["id"], **paper)

        # Forward direction (cited papers)
        queue = deque([(paper, 0)])
        while queue:
            current_paper, level = queue.popleft()

            if level < forward_levels:
                for cited_paper_id in current_paper.get("references", []):
                    cited_paper = self.find_paper_by_id(cited_paper_id)
                    if cited_paper and not citation_network.has_node(cited_paper_id):
                        citation_network.add_node(cited_paper_id, **cited_paper)
                        queue.append((cited_paper, level + 1))
                    citation_network.add_edge(current_paper["id"], cited_paper_id)

        # Backward direction (citing papers)
        queue = deque([(paper, 0)])
        while queue:
            current_paper, level = queue.popleft()

            if level < backward_levels:
                citing_papers = self.find_citing_papers(current_paper["id"])
                for citing_paper in citing_papers:
                    if citing_paper and not citation_network.has_node(citing_paper["id"]):
                        citation_network.add_node(citing_paper["id"], **citing_paper)
                        queue.append((citing_paper, level + 1))
                    citation_network.add_edge(citing_paper["id"], current_paper["id"])

        return citation_network


    # Custom layout function for visualizing the citation network
    def bidirectional_layered_layout(self, G, root):
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
    def bibliographic_coupling(self, G, X, Y):
        X_references = set(G.neighbors(X))
        Y_references = set(G.neighbors(Y))
        return len(X_references.intersection(Y_references))


    def co_citation(self, G, X, Y):
        X_cited_by = set(G.predecessors(X))
        Y_cited_by = set(G.predecessors(Y))
        return len(X_cited_by.intersection(Y_cited_by))


    # Functions for calculating candidate score

    def candidate_score(self, G, paper_of_interest, paper):
        if nx.has_path(self.citation_network, source=paper_of_interest, target=paper):
            distance = nx.shortest_path_length(self.citation_network, source=paper_of_interest, target=paper)
        else:
            distance = float('inf')  # or some other large number indicating a very long distance
        # distance = nx.shortest_path_length(G, source=paper_of_interest, target=paper)
        bc = self.bibliographic_coupling(G, paper_of_interest, paper)
        cc = self.co_citation(G, paper_of_interest, paper)
        return (bc + cc) / distance


    def get_candidates(self, id: str) -> list:
        paper = self.papers_dict[id]  # Replace 110 with the id of the paper you want to start with
        self.citation_network = self.build_citation_network(paper)
        paper_of_interest = paper["id"]
        # print(citation_network.nodes)
        # print(citation_network.edges)

        print("paper of interest")
        print(paper)
        candidate_papers = []

        for other_paper in self.citation_network.nodes:
            if other_paper != paper_of_interest:
                bc = self.bibliographic_coupling(self.citation_network, paper_of_interest, other_paper)
                cc = self.co_citation(self.citation_network, paper_of_interest, other_paper)
                score = self.candidate_score(self.citation_network, paper_of_interest, other_paper)
                if nx.has_path(self.citation_network, source=paper_of_interest, target=other_paper):
                    distance = nx.shortest_path_length(self.citation_network, source=paper_of_interest, target=other_paper)
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
        return candidate_papers


    def rank_papers(self, candidate_papers, paper_of_interest, top_n=10):
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
                "eigenvector": eigenvector,
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

    def recommend(self, paper: str):
        candidate_papers = self.get_candidates(paper)

        top_papers = self.rank_papers(candidate_papers, paper)
        output = []
        for rank, paper in enumerate(top_papers, start=1):
            id = paper['paper']
            paper_info = self.papers_dict.get(id, None)
            if paper_info:
                title = paper_info.get('paper title', "Title not found")
                paper_info["score"] = (len(top_papers) - paper["average_rank"]) / len(top_papers)
                output.append(paper_info)
            else:
                title = "Paper information not found"
            print(f"Rank {rank} - Paper ID: {id}, Title: {title}")

        return output, top_papers

if __name__ == '__main__':
    import json
    import pprint
    import pymongo

    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
    db = client["Aminer"]  # Replace with your database name
    collection = db["papers"]  # Replace with your collection name

    # Query the MongoDB collection to retrieve the data
    data = list(collection.find())

    # Create a dictionary to store the papers indexed by their ids
    papers_dict = {paper["id"]: paper for paper in data}

    system = Baseline2(data)

    papers, top = system.recommend("322302")

    pprint.pprint(papers)
    pprint.pprint(top)
