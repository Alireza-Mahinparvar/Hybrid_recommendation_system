from crawler import Crawler
from collaborative_filtering import CollaborativeFilteringModule
from content_based import ContentBasedModule

import pymongo
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, average_precision_score

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

        sorted_candidates = []
        for candidate in candidates:
            print(f"Calculating content-based score for paper {candidate}")
            content_score = self.content_based.cosine_simi(query_tf, self.content_based.term_freq(self.refs[candidate]))

            print(f"Calculating collaborative filtering score for paper {candidate}")
            collab_score = self.collab_filter.get_total_score(paper, candidate)

            candidates[candidate]['score'] = self._normalize_score(content_score, collab_score)
            sorted_candidates.append(candidates[candidate])

        sorted_candidates = sorted(sorted_candidates, key=lambda x: x['score'], reverse=True)
        recommended_papers = sorted_candidates[:min(5, len(candidates))]

        return recommended_papers

if __name__ == '__main__':
    import json
    import pprint

    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
    db = client["Aminer"]  # Replace with your database name
    collection = db["Aminer_full_data"]  # Replace with your collection name

    # Query the MongoDB collection to retrieve the data
    data = list(collection.find())

    # Create a dictionary to store the papers indexed by their ids
    papers_dict = {paper["id"]: paper for paper in data}

    # Split data into train and test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    system = Recommender(papers_dict)

    # Initialize lists to store evaluation results
    precision_scores = []
    recall_scores = []
    map_scores = []

    for test_paper in test_data:
        actual = {test_paper["id"]}
        recommendations = {paper['id'] for paper in system.recommend(test_paper["id"])}

        # Calculate precision, recall, and MAP for each test paper individually
        true_positives = len(actual.intersection(recommendations))
        precision = true_positives / len(recommendations) if len(recommendations) > 0 else 0
        recall = true_positives / len(actual)
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
