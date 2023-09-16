from Baseline1.crawler import Crawler
from Baseline1.collaborative_filtering import CollaborativeFilteringModule
from Baseline1.content_based import ContentBasedModule

import pymongo

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
            print(f"Calculating content based score for paper {candidate}")
            content_score = self.content_based.cosine_simi(query_tf, self.content_based.term_freq(self.refs[candidate]))

            print(f"Calculating collaborative filtering score for paper {candidate}")
            collab_score = self.collab_filter.get_total_score(paper, candidate)

            candidates[candidate]['score'] = self._normalize_score(content_score, collab_score)
            sorted_candidates.append(candidates[candidate])

        sorted_candidates = sorted(sorted_candidates, key=lambda x: x['score'], reverse=True)
        return sorted_candidates[:min(10, len(candidates))]

if __name__ == '__main__':
    import json
    import pprint

    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
    db = client["Aminer"]  # Replace with your database name
    collection = db["papers"]  # Replace with your collection name

    # Query the MongoDB collection to retrieve the data
    data = list(collection.find())

    # Create a dictionary to store the papers indexed by their ids
    papers_dict = {paper["id"]: paper for paper in data}

    system = Recommender(papers_dict)

    pprint.pprint(system.recommend("322302"))
