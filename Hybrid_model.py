import pymongo
import networkx as nx
import matplotlib.pyplot as plt
import community
import torch
import torch.nn as nn
import re
import math
import nltk
import itertools
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords
import networkx as nx


class CitationNetworkBuilder:
    def __init__(self, papers_dict):
        self.papers_dict = papers_dict

    def create_citation_network(self, paper_id, papers_collection, layers_forward=5, layers_backward=5):
        G = nx.DiGraph()

        def explore_citations(current_paper_id, current_layer, direction='forward'):
            if current_layer == 0:
                return

            paper = papers_collection.find_one({"id": current_paper_id})
            if not paper:
                return

            current_node = paper['id']
            G.add_node(current_node)

            if direction == 'forward':
                for cited_paper_id in paper.get('references', []):
                    cited_paper = papers_collection.find_one({"id": cited_paper_id})
                    if cited_paper:
                        cited_node = cited_paper['id']
                        G.add_edge(current_node, cited_node)
                        explore_citations(cited_paper_id, current_layer - 1, 'forward')
            elif direction == 'backward':
                citing_papers = papers_collection.find({"references": current_paper_id})
                for citing_paper in citing_papers:
                    citing_node = citing_paper['id']
                    G.add_edge(citing_node, current_node)
                    explore_citations(citing_node, current_layer - 1, 'backward')

        explore_citations(paper_id, layers_forward, 'forward')
        explore_citations(paper_id, layers_backward, 'backward')

        return G

    # Algorithm 2
    def calculate_bc_cc(self, G):
        # Initialize BC and CC scores for each paper
        bc_scores = {paper: 0 for paper in G.nodes}
        cc_scores = {paper: 0 for paper in G.nodes}

        # Compute BC and CC scores for all pairs of nodes
        for X, Y in itertools.combinations(G.nodes, 2):
            # Papers cited by X and Y for BC
            citations_X = set(G.successors(X))
            citations_Y = set(G.successors(Y))
            # Papers citing X and Y for CC
            references_X = set(G.predecessors(X))
            references_Y = set(G.predecessors(Y))

            # Calculate intersection for BC and CC
            common_citations = citations_X.intersection(citations_Y)
            common_references = references_X.intersection(references_Y)

            # Update BC and CC scores
            bc_score = len(common_citations)
            cc_score = len(common_references)
            bc_scores[X] += bc_score
            bc_scores[Y] += bc_score
            cc_scores[X] += cc_score
            cc_scores[Y] += cc_score

        # Store BC and CC scores as node attributes
        for paper in G.nodes():
            G.nodes[paper]['bc_score'] = bc_scores[paper]
            G.nodes[paper]['cc_score'] = cc_scores[paper]
        return G

    def calculate_candidate_score(self, G):
        # Iterating over each node to calculate candidate score
        for node_id in G.nodes:
            # Retrieve the BC, CC, and distance for the node
            bc = G.nodes[node_id].get('bc_score', 0)  # Assuming default is 0 if not present
            cc = G.nodes[node_id].get('cc_score', 0)  # Assuming default is 0 if not present
            distance = G.nodes[node_id].get('distance')

            # Ensure the distance is not zero to avoid division by zero error
            if distance is not None and distance != 0:
                # Calculate candidate score
                candidate_score = (bc + cc) / distance
            else:
                # If the distance is zero, it usually means this is the source node.
                # You might want to handle this case as per your requirements.
                candidate_score = float('inf')  # or some other rule, like bc + cc

            # Set the candidate score as a node attribute
            nx.set_node_attributes(G, {node_id: {'candidate_score': candidate_score}})

        # Verify by printing the node attributes
        # for node_id, attributes in G.nodes(data=True):
        #     print(f"Node {node_id} has a candidate score of {attributes.get('candidate_score')}")

        return G

    ## Algorithm 3
    def calculate_centrality(self, G):
        # Calculate in-degree centrality for each node
        in_degree_centrality = nx.in_degree_centrality(G)

        # Store in-degree centrality as a node attribute
        nx.set_node_attributes(G, in_degree_centrality, 'in_degree_centrality')

        # Assuming you have a graph G
        closeness_centrality = nx.closeness_centrality(G)

        # To store the closeness centrality as a node attribute
        nx.set_node_attributes(G, closeness_centrality, 'closeness_centrality')

        betweenness_centrality = nx.betweenness_centrality(G, normalized=True)

        # Store betweenness centrality as a node attribute
        nx.set_node_attributes(G, betweenness_centrality, 'betweenness_centrality')

        # Calculate eigenvector centrality
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)

        # Store eigenvector centrality as a node attribute
        nx.set_node_attributes(G, eigenvector_centrality, 'eigenvector_centrality')

        return G

    def scale_and_rank(self, G):
        # Assuming G is your directed graph with nodes already having 'degree', 'betweenness', 'closeness', 'eigen' attributes
        # Initialize a dictionary to store the scaled ranks
        scaled_ranks = {'in_degree_centrality': {}, 'betweenness_centrality': {}, 'closeness_centrality': {},
                        'eigenvector_centrality': {}}

        # Scale each centrality measure and store the ranks
        for measure in scaled_ranks.keys():
            # Get a list of (node, centrality value) for each centrality measure
            centrality_values = [(node, data[measure]) for node, data in G.nodes(data=True)]
            # Sort based on centrality value
            centrality_values.sort(key=lambda x: x[1], reverse=True)
            # Assign ranks and scale them to (1:50)
            for rank, (node, _) in enumerate(centrality_values, start=1):
                scaled_ranks[measure][node] = 1 + 49 * ((rank - 1) / (len(G.nodes()) - 1))

        # Calculate the average rank for each node
        average_ranks = {}
        for node in G.nodes():
            rank_sum = sum(scaled_ranks[measure][node] for measure in scaled_ranks)
            average_ranks[node] = rank_sum / len(scaled_ranks)

        # Now, you might want to store the average rank back into the graph as a node attribute
        nx.set_node_attributes(G, average_ranks, 'average_rank')

        return G

    def select_candidates(self, G):
        # Retrieve all candidate scores that are not infinity.
        candidate_scores = [data['candidate_score'] for node, data in G.nodes(data=True) if
                            np.isfinite(data['candidate_score'])]

        # Calculate the 50th percentile as our threshold.
        threshold = np.percentile(candidate_scores, 50)
        print(f'Threshold: {threshold}')
        print(f'Initial number of nodes : {G.number_of_nodes()}')
        # Now, we filter out nodes that are below our threshold.
        nodes_to_remove = [node for node, data in G.nodes(data=True) if data['candidate_score'] < threshold]

        # Remove these nodes from the graph.
        G.remove_nodes_from(nodes_to_remove)
        print(f'Number of nodes after removing the ones below threshold: {G.number_of_nodes()}')
        # If you want to keep the removed nodes information, for instance, you can create a subgraph before removing.
        removed_nodes_subgraph = G.subgraph(nodes_to_remove).copy()
        # Now G has only the nodes above the threshold.
        return G


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
    def __init__(self, papers_dict, papers_collection, device='cuda'):
        self.papers_dict = papers_dict
        self.papers_collection = papers_collection
        self.device = device

        # Create an instance of the CitationNetworkBuilder class
        self.citation_builder = CitationNetworkBuilder(self.papers_dict)

        # Load the pre-trained content-based and collaborative filtering modules
        print("content based module creation")
        self.content_based_module = ContentBasedModule(self.papers_dict).to(self.device)

    def recommend(self, paper_id, top_k=10, penalty=10):
        print("finding subset")
        subset, _ = Crawler(self.papers_dict).get_subset(paper_id)
        self.citation_network = self.citation_builder.create_citation_network(paper_id, self.papers_collection, layers_forward=5, layers_backward=5)
        print("collab filtering module")
        # Create cooccurred and coccurring matrices
        self.collaborative_filtering_module = CollaborativeFilteringModule(subset).to(self.device)

        # generate set of papers to run scoring on
        candidates = set()
        for paper in subset:
            candidates.add(subset[paper]["id"])
        for node in self.citation_network.nodes:
            candidates.add(node)

        # generate scores for each candidate using baseline 1
        b1_similarity_scores = []
        for candidate in candidates:
            # Calculate the content-based similarity score
            content_based_similarity_score = self.content_based_module(paper_id, candidate)

            # Calculate the collaborative filtering similarity score
            collaborative_filtering_similarity_score = self.collaborative_filtering_module(paper_id, candidate)

            # combine scores
            total_score = content_based_similarity_score + collaborative_filtering_similarity_score

            b1_similarity_scores.append((candidate, total_score))

        # Sort the papers by baseline 1 similarity score
        b1_sorted_papers = sorted(b1_similarity_scores, key=lambda x: x[1], reverse=True)

        # baseline 2 actions
        G = self.citation_builder.calculate_bc_cc(self.citation_network)
        # get distance from paper of interest
        lengths = nx.single_source_shortest_path_length(G.to_undirected(), paper_id)

        # Store these distances as node attributes in the original directed graph
        nx.set_node_attributes(G, lengths, 'distance')
        G = self.citation_builder.calculate_candidate_score(G)
        G = self.citation_builder.select_candidates(G)
        # Algorithm 3
        G = self.citation_builder.calculate_centrality(G)
        G = self.citation_builder.scale_and_rank(G)

        # The graph G now has an 'average_rank' attribute for each node
        # Sort the nodes by the 'average_rank' attribute in descending order (lowest rank first)
        sorted_nodes = G.nodes(data=True)

        combined_dict = {}
        for i in range(len(b1_sorted_papers)):
            paper = b1_sorted_papers[i]
            combined_dict[paper[0]] = {"id": paper[0], "rank": i + penalty}
        for node in sorted_nodes:
            if node[0] not in combined_dict:
                combined_dict[node[0]] = {"id": node[0], "rank": node[1]["average_rank"] + penalty}
            else:
                combined_dict[node[0]]["rank"] = combined_dict[node[0]]["rank"] - penalty + node[1]["average_rank"]


        sorted_papers = sorted(list(combined_dict.values()), key=lambda x: x["rank"], reverse=False)
        # Get the top-k recommendations based on author ranking
        author_ranking_recommendations = []
        for paper in sorted_papers:
            paper_index = paper["id"]
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

    # Create a dictionary to store the papers indexed by their ids
    papers_dict = {paper["id"]: paper for paper in data}

    # Create an instance of the HybridRecommendationSystem class
    print("instantiating model")
    recommendation_system = HybridRecommendationSystem(papers_dict, collection)

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

