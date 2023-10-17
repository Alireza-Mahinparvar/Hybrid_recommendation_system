import pymongo
import networkx as nx
import matplotlib.pyplot as plt
import community
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from collections import deque
import pandas as pd
import re
import math
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
import networkx as nx


class CitationNetworkBuilder:
    def __init__(self, papers_dict, reverse_references):
        self.papers_dict = papers_dict
        self.reverse_references = reverse_references

    def find_paper_by_id(self, paper_id):
        return self.papers_dict.get(paper_id)

    def find_citing_papers(self, paper_id):
        citing_paper_ids = self.reverse_references.get(paper_id, [])
        return [self.find_paper_by_id(paper_id) for paper_id in citing_paper_ids]

    def build_citation_network(self, paper, forward_levels=5, backward_levels=5):
        paper_id = paper["id"]
        citation_network = nx.DiGraph()
        citation_network.add_node(paper_id, **paper)  # Use paper information as attributes

        def explore_citations(paper_id, level, direction):
            if level < forward_levels and direction == "forward":
                for cited_paper_id in paper.get("references", []):
                    if not citation_network.has_node(cited_paper_id):
                        cited_paper = self.papers_dict.get(cited_paper_id, {})
                        citation_network.add_node(cited_paper_id, **cited_paper)
                        citation_network.add_edge(paper_id, cited_paper_id)
                        explore_citations(cited_paper_id, level + 1, "forward")

            if level < backward_levels and direction == "backward":
                for citing_paper_id in self.reverse_references.get(paper_id, []):
                    if not citation_network.has_node(citing_paper_id):
                        citing_paper = self.papers_dict.get(citing_paper_id, {})
                        citation_network.add_node(citing_paper_id, **citing_paper)
                        citation_network.add_edge(citing_paper_id, paper_id)
                        explore_citations(citing_paper_id, level + 1, "backward")

        explore_citations(paper_id, 0, "forward")
        explore_citations(paper_id, 0, "backward")

        return citation_network

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

    def bibliographic_coupling(self, G, X, Y):
        X_references = set(G.neighbors(X))
        Y_references = set(G.neighbors(Y))
        return len(X_references.intersection(Y_references))

    def co_citation(self, G, X, Y):
        X_cited_by = set(G.predecessors(X))
        Y_cited_by = set(G.predecessors(Y))
        return len(X_cited_by.intersection(Y_cited_by))

    def get_top_recommendations(self, paper_of_interest, top_n=10):
        candidate_papers = []

        for other_paper in self.citation_network.nodes:
            if other_paper != paper_of_interest:
                if len(self.papers_dict[other_paper].get("references", [])) > 0 or len(
                        self.find_citing_papers(other_paper)) > 0:
                    bc = self.bibliographic_coupling(self.citation_network, paper_of_interest, other_paper)
                    cc = self.co_citation(self.citation_network, paper_of_interest, other_paper)
                    score = self.candidate_score(self.citation_network, paper_of_interest, other_paper)
                    if nx.has_path(self.citation_network, source=paper_of_interest, target=other_paper):
                        distance = nx.shortest_path_length(self.citation_network, source=paper_of_interest,
                                                           target=other_paper)
                    else:
                        distance = float('inf')  # or some other large number indicating a very long distance

                    candidate_papers.append({
                        "paper": other_paper,
                        "bibliographic_coupling": bc,
                        "co_citation": cc,
                        "score": score,
                        "distance": distance,
                    })

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

        # Get top recommendations using the rank_papers method
        top_recommendations = rank_papers(candidate_papers, top_n)
        return top_recommendations


from itertools import combinations


class CollaborativeFilteringModule(nn.Module):
    def __init__(self, refs):
        super(CollaborativeFilteringModule, self).__init__()
        self.refs = refs
        self.cooccurred = self.generate_cooccurred_matrix(refs)
        self.cooccurring = self.generate_cooccurring_matrix(refs)

    def generate_cooccurred_matrix(self, refs: dict) -> dict:
        """
        Generate cooccurred matrix for given reference relations
        :param refs: list of dictionaries that contain 'references' key
        :return: cooccurred matrix where matrix[i][j]=1 if they cooccurred
        """
        matrix = {}
        for paper in refs:
            if 'references' in refs[paper]:
                # look at each pair of referenced papers, papers cooccurred if they are cited by same paper
                for pair in combinations(refs[paper]['references'], 2):
                    if pair[0] not in matrix:
                        matrix[pair[0]] = {}
                    if pair[1] not in matrix:
                        matrix[pair[1]] = {}
                    matrix[pair[0]][pair[1]] = 1
                    matrix[pair[1]][pair[0]] = 1
        return matrix

    def generate_cooccurring_matrix(self, refs: dict) -> dict:
        """
        Generate cooccurring matrix for given reference relations
        :param refs: list of dictionaries that contain 'references' key
        :return: cooccurring matrix where matrix[i][j]=1 if they cooccurring
        """
        matrix = {}
        # look at all combinations f papers to see if they cite the same paper (cooccurring)
        for pair in combinations(list(refs.keys()), 2):
            if 'references' in refs[pair[0]] and 'references' in refs[pair[1]]:
                if any(i in refs[pair[0]]['references'] for i in refs[pair[1]]['references']):
                    if pair[0] not in matrix:
                        matrix[pair[0]] = {}
                    if pair[1] not in matrix:
                        matrix[pair[1]] = {}
                    matrix[pair[0]][pair[1]] = 1
                    matrix[pair[1]][pair[0]] = 1
        return matrix

    def get_cooccurred_score(self, paper1: str, paper2: str) -> float:
        """
        Calculates cooccurred score
        :param paper1: index of paper 1
        :param paper2: index of paper 2
        :return: cooccurred score
        """
        row1 = self.cooccurred.get(paper1, {})
        row2 = self.cooccurred.get(paper2, {})
        j11 = 0
        j10 = 0
        for i in row1:
            r1 = row1.get(i, 0)
            r2 = row2.get(i, 0)
            if r1 == 1 and r2 == 1:
                j11 += 1
            elif r1 == 1 ^ r2 == 1:
                j10 += 1
        total = (j11 + j10)
        return j11 / total if total > 0 else 0

    def get_cooccurring_score(self, paper1: str, paper2: str) -> float:
        """
        Calculates cooccurring score
        :param paper1: index of paper 1
        :param paper2: index of paper 2
        :return: cooccurring score
        """
        row1 = self.cooccurring.get(paper1, {})
        row2 = self.cooccurring.get(paper2, {})
        j11 = 0
        j10 = 0
        for i in row1:
            r1 = row1.get(i, 0)
            r2 = row2.get(i, 0)
            if r1 == 1 and r2 == 1:
                j11 += 1
            elif r1 == 1 ^ r2 == 1:
                j10 += 1
        total = (j11 + j10)
        return j11 / total if total > 0 else 0

    def forward(self, paper1, paper2):
        # Calculate cooccurred and cooccurring scores between two papers
        cooccurred_score = self.get_cooccurred_score(paper1, paper2)
        cooccurring_score = self.get_cooccurring_score(paper1, paper2)
        return (cooccurred_score + cooccurring_score) / 2


class Crawler:
    def __init__(self, refs: dict = {}):
        self.refs = refs

    def get_subset(self, query: str):
        subset = {}
        candidates = {}
        subset[query] = self.refs[query]

        # get papers citing query and their references
        for id in self.refs:
            paper = self.refs[id]
            if 'references' in paper:
                if query in paper['references']:
                    if id not in subset:
                        subset[id] = paper
                    for ref in paper['references']:
                        if ref not in subset:
                            subset[ref] = self.refs[ref]
                            candidates[ref] = self.refs[ref]

        # get query's references and those that cited query's references
        if "references" in self.refs[query]:
            for ref in self.refs[query]['references']:
                if ref not in candidates:
                    subset[ref] = self.refs[ref]
                    candidates[ref] = self.refs[ref]
                for id in self.refs:
                    if id != query:
                        paper = self.refs[id]
                        if 'references' in paper:
                            if ref in paper['references']:
                                if id not in subset:
                                    subset[id] = paper
                                if id not in candidates:
                                    candidates[id] = paper

        return subset, candidates


stop_words = set(stopwords.words('english'))


class ContentBasedModule(nn.Module):
    def __init__(self, papers_dict):
        super(ContentBasedModule, self).__init__()
        self.papers_dict = papers_dict

    def term_freq(self, paper_id) -> dict:
        """
        Calculates term frequency for given paper
        :param paper: dict containing 'title' and/or 'abstract'
        :return: dictionairy containing word frequencies of form
        {
            'word': term_frequency
        }
        """
        paper = self.papers_dict[paper_id]
        f_dict = {}
        count = 0
        fields = ['paper title', 'abstract', 'keywords']
        for field in fields:
            if field in paper:
                string = paper[field]
                # Replace all single characters with a space
                string = re.sub(r'\b[a-zA-Z]\b', ' ', string)
                # Replace all double spaces with one space
                string = re.sub(' +', ' ', string)
                # Remove leading and trailing spaces
                string = string.strip().lower()
                words = list(string.split(" "))
                for word in words:
                    if word not in stop_words:
                        count += 1
                        if word not in f_dict:
                            f_dict[word] = 1
                        elif word in f_dict:
                            f_dict[word] += 1
        for word in f_dict:
            f_dict[word] = f_dict[word] / count
        return f_dict

    def cosine_similarity(self, paper1: dict, paper2: dict) -> float:
        """
        Calculate cosine simialrity of 2 word frequency dictionaries
        :param paper1: word frequencies of paper 1
        :param paper2: word frequencies of paper 2
        :return: cosine similarity score
        """
        word_list1 = list(paper1.keys())
        word_list2 = set(paper2.keys())
        dot_prod = 0
        dist1 = 0
        dist2 = 0
        for word in word_list1:
            if word in word_list2:
                dot_prod += paper1[word] * paper2[word]
            dist1 += paper1[word] * paper1[word]
        dist1 = math.sqrt(dist1)
        for word in word_list2:
            dist2 += paper2[word] * paper2[word]
        dist2 = math.sqrt(dist2)
        return dot_prod / (dist1 + dist2)

    def forward(self, paper1_id, paper2_id):
        # Calculate cosine similarity between two papers
        similarity = self.cosine_similarity(self.term_freq(paper1_id), self.term_freq(paper2_id))
        return similarity


class HybridRecommendationSystem:
    def __init__(self, papers_dict, reverse_references, device='cuda'):
        self.papers_dict = papers_dict
        self.reverse_references = reverse_references
        self.device = device

        # Create an instance of the CitationNetworkBuilder class
        citation_builder = CitationNetworkBuilder(self.papers_dict, self.reverse_references)

        # Call the build_citation_network function
        print("building citation network")
        self.citation_network = citation_builder.build_citation_network(self.papers_dict["556798"])
        adjacency_matrix = nx.adjacency_matrix(self.citation_network, weight='weight')
        self.citation_similarity_matrix = adjacency_matrix.toarray()

        # Convert the directed citation network into an undirected graph
        undirected_citation_network = self.citation_network.to_undirected()

        # Apply the Louvain community detection algorithm to the undirected graph
        print("partition time")
        self.communities = community.best_partition(undirected_citation_network)

        # Load the pre-trained content-based and collaborative filtering modules
        print("content based module creation")
        self.content_based_module = ContentBasedModule(self.papers_dict).to(self.device)

    def recommend(self, paper_id, top_k=10):
        print("finding subset")
        subset, _ = Crawler(self.papers_dict).get_subset(paper_id)
        print("collab filtering module")
        # Create cooccurred and coccurring matrices
        self.collaborative_filtering_module = CollaborativeFilteringModule(subset).to(self.device)

        # generate set of papers to run scoring on
        candidates = set()
        for paper in subset:
            candidates.add(subset[paper]["id"])
        for node in self.citation_network.nodes:
            candidates.add(node)

        # generate scores for each candidate
        hybrid_similarity_scores = []
        for candidate in candidates:
            # Calculate the content-based similarity score
            content_based_similarity_score = self.content_based_module(paper_id, candidate)

            # Calculate the collaborative filtering similarity score
            collaborative_filtering_similarity_score = self.collaborative_filtering_module(paper_id, candidate)

            # combine scores
            total_score = content_based_similarity_score + collaborative_filtering_similarity_score

            hybrid_similarity_scores.append((candidate, total_score))

        # Sort the papers by hybrid similarity score
        sorted_papers = sorted(hybrid_similarity_scores, key=lambda x: x[1], reverse=True)

        # Get the top-k recommendations based on author ranking
        author_ranking_recommendations = []
        for paper_index, similarity_score in sorted_papers:
            if paper_index != paper_id:  # Exclude the target paper itself
                paper = self.papers_dict.get(paper_index)
                if paper:
                    authors = paper.get('authors', [])
                    author_rank = len(authors)  # Author ranking based on the number of authors
                    author_ranking_recommendations.append((paper_index, author_rank))
                    if len(author_ranking_recommendations) >= top_k:
                        break

        return [paper_index for paper_index, _ in author_ranking_recommendations]


if __name__ == "__main__":
    # Connect to MongoDB
    print("connecting to mongodb")
    client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
    db = client["Aminer"]  # Replace with your database name
    collection = db["papers"]  # Replace with your collection name

    # Query the MongoDB collection to retrieve the data
    data = list(collection.find())

    print("dataset refinement")
    # Create a dictionary to store the papers indexed by their ids
    papers_dict = {paper["id"]: paper for paper in data}
    reverse_references = {}
    for paper in data:
        for ref_id in paper.get("references", []):
            if ref_id not in reverse_references:
                reverse_references[ref_id] = []
            reverse_references[ref_id].append(paper["id"])

    # Create an instance of the HybridRecommendationSystem class
    print("instantiating model")
    recommendation_system = HybridRecommendationSystem(papers_dict, reverse_references)

    # Specify the paper_id for which you want to get recommendations
    target_paper_id = "556798"

    # Get recommendations for the specified paper_id
    print("recommending")
    top_k_recommendations = recommendation_system.recommend(target_paper_id, top_k=10)

    print("Top 10 recommendations for paper with ID", target_paper_id)
    for paper_id in top_k_recommendations:
        paper = papers_dict.get(paper_id)
        if paper:
            print(f"Paper ID: {paper_id}, Title: {paper.get('paper title', 'N/A')}")

