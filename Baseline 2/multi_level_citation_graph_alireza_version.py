import pymongo
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import community

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
db = client["Aminer"]  # Replace with your database name
collection = db["Aminer_full_data"]  # Replace with your collection name

# Query the MongoDB collection to retrieve the data
data = list(collection.find())

# Create a dictionary to store the papers indexed by their ids
papers_dict = {paper["id"]: paper for paper in data}

reverse_references = {}
for paper in data:
    for ref_id in paper.get("references", []):
        if ref_id not in reverse_references:
            reverse_references[ref_id] = []
        reverse_references[ref_id].append(paper["id"])

# Helper functions
def find_paper_by_id(paper_id):
    return papers_dict.get(paper_id)

def find_citing_papers(paper_id):
    citing_paper_ids = reverse_references.get(paper_id, [])
    return [find_paper_by_id(paper_id) for paper_id in citing_paper_ids]

# Main function to build citation network
def build_citation_network(paper, forward_levels=5, backward_levels=5):
    paper_id = paper["id"]
    citation_network = nx.DiGraph()
    citation_network.add_node(paper_id, **paper)  # Use paper information as attributes

    def explore_citations(paper_id, level, direction):
        if level < forward_levels and direction == "forward":
            for cited_paper_id in paper.get("references", []):
                if not citation_network.has_node(cited_paper_id):
                    cited_paper = papers_dict.get(cited_paper_id, {})
                    citation_network.add_node(cited_paper_id, **cited_paper)
                    citation_network.add_edge(paper_id, cited_paper_id)
                    explore_citations(cited_paper_id, level + 1, "forward")

        if level < backward_levels and direction == "backward":
            for citing_paper_id in reverse_references.get(paper_id, []):
                if not citation_network.has_node(citing_paper_id):
                    citing_paper = papers_dict.get(citing_paper_id, {})
                    citation_network.add_node(citing_paper_id, **citing_paper)
                    citation_network.add_edge(citing_paper_id, paper_id)
                    explore_citations(citing_paper_id, level + 1, "backward")

    explore_citations(paper_id, 0, "forward")
    explore_citations(paper_id, 0, "backward")

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
paper = papers_dict['556798']  # Replace 110 with the id of the paper you want to start with
citation_network = build_citation_network(paper)
paper_of_interest = paper["id"]

print("paper of interest")
print(paper)
candidate_papers = []

# Draw and display the graph
pos = bidirectional_layered_layout(citation_network, paper["id"])
plt.figure(figsize=(12, 12))
nx.draw(citation_network, pos, node_size=50, font_size=8, with_labels=True, node_color="skyblue", edge_color="gray", arrows=True)
plt.title("Citation Network")
plt.savefig("citation_network.png", dpi=300, bbox_inches="tight")
plt.show()



for other_paper in citation_network.nodes:
    if other_paper != paper_of_interest:
        # Example usage to filter candidate papers

        if len(papers_dict[other_paper].get("references", [])) > 0 or len(find_citing_papers(other_paper)) > 0:
            

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




def rank_papers(candidate_papers, top_n= 10):
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

    centrality_df["average_rank"] = centrality_df[["degree_rank", "closeness_rank", "betweenness_rank", "eigenvector_rank"]].mean(axis=1)
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
        title = paper_info.get('paper title', "Title not found")
    else:
        title = "Paper information not found"
    print(f"Rank {rank} - Paper ID: {id}, Title: {title}")

"""
User Studies Function: This function replicates a user study in which consumers are given a list of suggested scholarly articles and asked 
to comment on the applicability of each article. It cycles through the suggested papers, showing the users their names and abstracts while 
also gathering user feedback. Statistics on the number of pertinent and irrelevant papers are computed based on the feedback. Additionally, 
precision, recall, and the F1-score are generated as evaluation metrics if there exist pertinent papers. In the absence of a ground truth dataset, 
this function is a useful tool for evaluating the user-perceived relevance of suggested papers.
"""
def user_studies(recommended_papers):
    print("User Studies of Recommended Papers:")
    
    user_feedback = []
    
    for rank, paper in enumerate(recommended_papers, start=1):
        paper_id = paper['paper']
        paper_info = papers_dict.get(paper_id, None)
        if paper_info:
            title = paper_info.get('paper title', "Title not found")
            abstract = paper_info.get('abstract', "Abstract not found")
        else:
            title = "Paper information not found"
            abstract = "Abstract not found"
        
        print(f"Rank {rank} - Paper ID: {paper_id}")
        print(f"Title: {title}")
        print(f"Abstract: {abstract}")
        
        # Prompt the user for feedback
        feedback = input("Is this paper relevant? (yes/no): ").strip().lower()
        
        # Process user feedback
        if feedback == 'yes':
            relevance_score = 1
        elif feedback == 'no':
            relevance_score = 0
        else:
            relevance_score = None
        
        # Store the user's feedback and relevance score
        user_feedback.append({
            'paper_id': paper_id,
            'feedback': feedback,
            'relevance_score': relevance_score
        })
        print("\n---\n")
    
    # Analyze the collected user feedback
    relevant_papers = [feedback['paper_id'] for feedback in user_feedback if feedback['relevance_score'] == 1]
    irrelevant_papers = [feedback['paper_id'] for feedback in user_feedback if feedback['relevance_score'] == 0]
    
    num_relevant = len(relevant_papers)
    num_irrelevant = len(irrelevant_papers)
    total_papers = len(recommended_papers)
    
    print("User Study Summary:")
    print(f"Total Recommended Papers: {total_papers}")
    print(f"Number of Relevant Papers: {num_relevant}")
    print(f"Number of Irrelevant Papers: {num_irrelevant}")
    
    if num_relevant + num_irrelevant > 0:
        # Calculate precision, recall, and F1-score (use 0 if there are no relevant papers)
        precision = num_relevant / (num_relevant + num_irrelevant)
        recall = num_relevant / (num_relevant + num_irrelevant)
        f1 = 2 * (precision * recall) / (precision + recall)
        
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")
    else:
        print("Precision, Recall, and F1-score cannot be calculated because there are no relevant papers.")
    
    return user_feedback, precision, recall, f1


user_studies(top_papers)



undirected_citation_network = citation_network.to_undirected()

# Perform community detection using the Louvain algorithm
partition = community.best_partition(undirected_citation_network)
#partition = community.best_partition(citation_network)

# Visualize the communities
pos = bidirectional_layered_layout(citation_network, paper_of_interest)
plt.figure(figsize=(12, 12))
cmap = plt.get_cmap('viridis', max(partition.values()) + 1)
nx.draw(citation_network, pos, node_size=50, font_size=8, with_labels=True, node_color=list(partition.values()), cmap=cmap, edge_color="gray", arrows=True)
plt.title("Citation Network with Communities (Louvain)")
plt.savefig("citation_network_communities.png", dpi=300, bbox_inches="tight")
plt.show()

# Evaluate the quality and relevance of communities
def evaluate_communities(partition, papers_dict):
    community_sizes = {}
    
    for paper_id, community_id in partition.items():
        if community_id not in community_sizes:
            community_sizes[community_id] = []
        community_sizes[community_id].append(paper_id)
    
    # Print the number of papers in each community
    print("\nCommunity Sizes:")
    for community_id, papers in community_sizes.items():
        print(f"Community {community_id}: {len(papers)} papers")


evaluate_communities(partition, papers_dict)
