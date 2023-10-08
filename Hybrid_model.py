import pymongo
import networkx as nx
from collections import deque
import matplotlib.pyplot as plt
import pandas as pd
import community
import sys

# Use a raw string to specify the file path
#sys.path.append(r'C:\Users\alire\Hybrid_recommendation_system-main\Baseline2')
from multi_level_citation_graph_alireza_version import build_citation_network
#sys.path.append(r'C:\Users\alire\Hybrid_recommendation_system-main\Baseline1')
from recommender import user_studies
from content_based import ContentBasedModule
from crawler import Crawler
from collaborative_filtering import CollaborativeFilteringModule
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.data import Data, DataLoader
import torch.optim as optim



# Define neural network architecture for the hybrid model
class RecommendationCommunityModel(nn.Module):
        def __init__(self, num_nodes, num_features, num_classes, hidden_dim):
            super(RecommendationCommunityModel, self).__init__()
        
            # Define the layers of  neural network here
            self.fc1 = nn.Linear(num_features, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

def extract_title_features(title):
    # Initialize title features dictionary
    title_features = {}
    
    title_length = len(title)
    title_features['title_length'] = title_length
    
    # Feature: Word count in the title
    title_words = title.split()
    title_word_count = len(title_words)
    title_features['title_word_count'] = title_word_count
    
    # You can add more title features as needed
    
    return title_features

def extract_abstract_features(abstract):
    # Initialize abstract features dictionary
    abstract_features = {}
    
    # Feature: Length of the abstract
    abstract_length = len(abstract)
    abstract_features['abstract_length'] = abstract_length
    
    # Feature: Word count in the abstract
    abstract_words = abstract.split()
    abstract_word_count = len(abstract_words)
    abstract_features['abstract_word_count'] = abstract_word_count
    
    # You can add more abstract features as needed
    
    return abstract_features

class Recommender:
    def __init__(self, refs: dict):
        self.refs = refs
        self.crawler = Crawler(refs)
        self.content_based = ContentBasedModule()
        self.collab_filter = None

    def _normalize_score(self, content_score: float, collab_score: float) -> float:
        normalized_score = (content_score + collab_score) / 2
        return normalized_score

    def recommend(self, paper: str) -> list:
        print(f"grabbing subset surrounding paper {paper}")
        subset, candidates = self.crawler.get_subset(paper)

        print(f"Calculating term frequency for paper {paper}")
        query_tf = self.content_based.term_freq(self.refs[paper])

        print(f"Calculating citation relations for subset")
        self.collab_filter = CollaborativeFilteringModule(subset)

        recommendations = []
        for candidate in candidates:
            print(f"Calculating content-based score for paper {candidate}")
            content_score = self.content_based.cosine_simi(query_tf, self.content_based.term_freq(self.refs[candidate]))

            print(f"Calculating collaborative filtering score for paper {candidate}")
            collab_score = self.collab_filter.get_total_score(paper, candidate)

            # Calculate the similarity score and include it in the result
            similarity_score = self._normalize_score(content_score, collab_score)

            candidates[candidate]['score'] = similarity_score  # Include similarity score
            recommendations.append(candidates[candidate])

        # Sort recommendations by score
        sorted_recommendations = sorted(recommendations, key=lambda x: x['score'], reverse=True)
        return sorted_recommendations[:min(10, len(candidates))]

if __name__ == '__main__':
    # Connect to MongoDB for the citation network data
    client = pymongo.MongoClient("mongodb://localhost:27017")
    db = client["Aminer"]
    collection = db["Aminer_full_data"]
    data = list(collection.find())
    papers_dict = {paper["id"]: paper for paper in data}
    
    # Define paper of interest
    paper_id = '556798'
    paper = papers_dict[paper_id]
    
    if paper:
    # Extract relevant features from the paper
        title = paper.get("paper title", "")
        abstract = paper.get("abstract", "")
        authors = paper.get("authors", [])
        affiliations = paper.get("affiliations", [])
        year = paper.get("year", "")
        publication_venue = paper.get("publication venue", "")
        references = paper.get("references", [])

    # Extract additional features as needed
    # For this example, we'll assume 'title_features' and 'abstract_features' are placeholders
        title_features = extract_title_features(title)
        abstract_features = extract_abstract_features(abstract)

    # Create the input_recommendation_data dictionary with all features
        input_recommendation_data = {
            'title_features': title_features,
            'abstract_features': abstract_features,
            'authors': authors,
            'affiliations': affiliations,
            'year': year,
            'publication_venue': publication_venue,
            'references': references,
        }

    # Now, input_recommendation_data contains all the extracted features
        print(input_recommendation_data)
    else:
        print(f"Paper with ID {paper_id} not found.")

    # Build the citation network
    citation_network = build_citation_network(paper)
    
    # Create an edge index and gather citation relationships
    edge_index = []
    for paper in data:
        paper_id = paper["id"]
        references = paper.get("references", [])
        for ref_id in references:
            edge_index.append((paper_id, ref_id))

    citation_network.add_edges_from(edge_index)

    citation_network_undirected = citation_network.to_undirected()


    partition = community.best_partition(citation_network_undirected )

    # Retrieve community labels for each paper
    community_labels = [partition[paper_id] for paper_id in citation_network.nodes]
    
    
    recommendation_system = Recommender(papers_dict)

        # Get recommendations for the target paper (including similarity scores)
    recommendations = recommendation_system.recommend(paper_id)

    author_encoder = OneHotEncoder(sparse=False)
# Flatten the list of authors and reshape it into a 2D array with a single feature
    author_encodings = author_encoder.fit_transform([[author] for rec in recommendations for author in rec['authors']])

# Create a label encoder for titles
    title_encoder = OneHotEncoder(sparse=False)
# Flatten the list of paper titles and reshape it into a 2D array with a single feature
    title_encodings = title_encoder.fit_transform([[title] for rec in recommendations for title in rec['paper title']])


# Convert the one-hot encoded features to PyTorch tensors
    author_one_hot = torch.tensor(author_encodings, dtype=torch.float32)
    title_one_hot = torch.tensor(title_encodings, dtype=torch.float32)
    # Make sure author_one_hot and title_one_hot have the same number of rows (should match the number of recommendations)
    num_recommendations = len(recommendations)
    author_one_hot = author_one_hot[:num_recommendations, :]
    title_one_hot = title_one_hot[:num_recommendations, :]

# Combine one-hot encoded features with similarity scores and other numeric features
    your_recommendation_features = torch.cat([
        torch.tensor([rec['score'] for rec in recommendations], dtype=torch.float32).unsqueeze(1),
        author_one_hot,
        title_one_hot,
    # ... other numeric features (e.g., input_recommendation_data['title_features'], etc.) ...
    ], dim=1)
    your_recommendation_features = torch.tensor(your_recommendation_features, dtype=torch.float32)

    # Create an instance of the neural network model
    unique_years = set()

    # Loop through your dataset and add each paper's year to the set
    for paper in data:
        year = paper.get("year", "")  # Get the year or an empty string if it's not available
        if year:
            unique_years.add(year)

    # Calculate the number of unique years
    num_classes = len(unique_years)
    num_nodes = len(papers_dict)
    num_features = 3
    hidden_dim = 32
    model = RecommendationCommunityModel(num_nodes, num_features, num_classes, hidden_dim)

    # Define loss functions and optimizer
    community_criterion = nn.CrossEntropyLoss()
    recommendation_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Prepare data and create DataLoader
    data = Data(x=your_recommendation_features, edge_index=edge_index, y=community_labels)
    batch_size = 32
    loader = DataLoader([data], batch_size=batch_size, shuffle=True)

    # Training loop (customize as needed)
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_data in loader:
            optimizer.zero_grad()
            community_predictions, recommendation_predictions = model(batch_data.x, batch_data.edge_index)
            
            # Compute losses for both tasks
            community_loss = community_criterion(community_predictions, batch_data.y)
            recommendation_loss = recommendation_criterion(recommendation_predictions, batch_data.x)
            
            # Combine the losses with a weighting factor
            total_loss = community_loss + 0.1 * recommendation_loss
            
            total_loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{epochs}] Loss: {total_loss.item()}')

    # Use the model for community detection and recommendation refinement
    
    community_embeddings, _ = model(input_recommendation_data, edge_index)
    predicted_communities = torch.argmax(community_embeddings, dim=1)

    # Example usage for recommendation refinement
    _, recommendation_embeddings = model(input_recommendation_data, edge_index)

