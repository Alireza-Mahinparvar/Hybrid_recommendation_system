import networkx as nx
import matplotlib.pyplot as plt
from pymongo import MongoClient
from pyvis.network import Network
import json
import itertools
import numpy as np

# Algorithm 1
def create_citation_network(paper_id,papers_collection, layers_forward=5, layers_backward=5):    
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
                    explore_citations(cited_paper_id, current_layer-1, 'forward')
        elif direction == 'backward':
            citing_papers = papers_collection.find({"references": current_paper_id})
            for citing_paper in citing_papers:
                citing_node = citing_paper['id']
                G.add_edge(citing_node, current_node)
                explore_citations(citing_node, current_layer-1, 'backward')

    explore_citations(paper_id, layers_forward, 'forward')
    explore_citations(paper_id, layers_backward, 'backward')
    
    return G

def visualize_network(G):
    plt.figure(figsize=(20, 10))
    
    # Define position for each node
    pos = {node: (0, idx) for idx, node in enumerate(G.nodes)}
    
    # Adjust the x-coordinates to position nodes in a left-to-right layout
    for node in G.nodes:
        successors = list(G.successors(node))
        predecessors = list(G.predecessors(node))
        if successors:
            # Position the cited papers to the right
            for succ_node in successors:
                pos[succ_node] = (pos[node][0] + 1, pos[succ_node][1])
        if predecessors:
            # Position the citing papers to the left
            for pred_node in predecessors:
                pos[pred_node] = (pos[node][0] - 1, pos[pred_node][1])
                
    nx.draw(G, pos, with_labels=True, font_size=10, node_size=2000, node_color="skyblue", font_color="black", alpha=0.5, linewidths=0.3)
    plt.show()

def visualize_interactive_network(G):
    # Initialize the PyVis Network
    nt = Network(notebook=True, height='800px', width='100%', directed=True)
    
    # Add nodes and edges to the network
    nt.from_nx(G)
    
    # Customize options for a cleaner look
    options = {
        'nodes': {
            'font': {'size': 14},
            'shape': 'dot',
            'size': 20,
        },
        'edges': {
            'arrows': 'to',
            'width': 1.0,
            'color': {'inherit': True},
            'smooth': {'type': 'continuous'}
        },
        # 'layout': {
        #     'hierarchical': {
        #         'enabled': True,
        #         'direction': 'LR',  # Left to Right
        #         'sortMethod': 'directed',
        #     }
        # },
        # 'physics': {
        #     'enabled': False,  # Disable physics to maintain the hierarchical layout
        # },
        'physics': {
            'solver': 'forceAtlas2Based',
            'timestep': 0.5,
            'stabilization': {'iterations': 150},
        },
        'interaction': {
            'zoomView': True  # Allow zooming
        }
    }

    # Convert options to JSON string
    options_json = json.dumps(options)
    
    nt.set_options(options_json)
    nt.show("citation_network.html")

# Algorithm 2
def calculate_bc_cc(G):
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
    
def calculate_candidate_score(G):
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
def calculate_centrality(G):
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

def scale_and_rank(G):
    # Assuming G is your directed graph with nodes already having 'degree', 'betweenness', 'closeness', 'eigen' attributes
    # Initialize a dictionary to store the scaled ranks
    scaled_ranks = {'in_degree_centrality': {}, 'betweenness_centrality': {}, 'closeness_centrality': {}, 'eigenvector_centrality': {}}

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

def select_candidates(G):
    # Retrieve all candidate scores that are not infinity.
    candidate_scores = [data['candidate_score'] for node, data in G.nodes(data=True) if np.isfinite(data['candidate_score'])]

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

if __name__ =='__main__':
    client = MongoClient('mongodb://localhost:27017/')
    db = client['Aminer']
    papers_collection = db['Paper']
    paper_id_of_interest = "1292825"

    # Algorithm 1
    G = create_citation_network(paper_id_of_interest, papers_collection, layers_forward=5, layers_backward=5)

    # Visualize the citation network
    # visualize_interactive_network(G)

    # Algorithm 2
    G = calculate_bc_cc(G)
    # get distance from paper of interest
    lengths = nx.single_source_shortest_path_length(G.to_undirected(), paper_id_of_interest)

    # Store these distances as node attributes in the original directed graph
    nx.set_node_attributes(G, lengths, 'distance')
    G = calculate_candidate_score(G)
    G = select_candidates(G)
    # Algorithm 3
    G = calculate_centrality(G)
    G = scale_and_rank(G)

    # The graph G now has an 'average_rank' attribute for each node
    # Sort the nodes by the 'average_rank' attribute in descending order (lowest rank first)
    sorted_nodes = sorted(G.nodes(data=True), key=lambda x: x[1]['average_rank'])

    # Extract the top 10 paper IDs
    top_10_paper_ids = [node[0] for node in sorted_nodes[:10]]

    print("Top 10 papers based on average rank:")
    for paper_id in top_10_paper_ids:
        paper = papers_collection.find_one({"id": paper_id})
        titile = paper.get("paper title")
        print(f"{paper_id} : {titile} ")




