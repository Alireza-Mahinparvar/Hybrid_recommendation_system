from Baseline1.crawler import Crawler
from Baseline1.collaborative_filtering import CollaborativeFilteringModule
from Baseline1.content_based import ContentBasedModule

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
    import pickle

    RUN_MONGO = False # set to true for first runs

    if RUN_MONGO:
        # Connect to MongoDB
        client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
        db = client["PaperRecommender"]  # Replace with your database name

        print("grabbing ground truth")
        ground_truth_collection = db["ground_truth"]
        truth_data = list(ground_truth_collection.find())
        ground_truth = {}
        for truth in truth_data:
            author = truth["author"]
            recommendation = truth["recommendation"]
            if author not in ground_truth:
                ground_truth[author] = []
            ground_truth[author].append(recommendation)
        pickle.dump(ground_truth, open("ground_truth.p", "wb"))

        print("processing papers data")
        paper_collection = db["papers"]  # Replace with your collection name
        # Query the MongoDB collection to retrieve the data
        data = list(paper_collection.find())
        # Create a dictionary to store the papers indexed by their ids
        papers_dict = {paper["id"]: paper for paper in data}

        print("processing author data")
        author_collection = db["authors"]
        author_data = list(author_collection.find())
        author_dict = {}
        for authorship in author_data:
            author = authorship["author"]
            paper = authorship["paper"]
            if author not in author_dict:
                author_dict[author] = []
            author_dict[author].append(paper)
            if paper in papers_dict:
                if "author" not in papers_dict[paper]:
                    papers_dict[paper]["author"] = [author]
                else:
                    papers_dict[paper]["author"].append(author)
        pickle.dump(author_dict, open("author_dict.p", "wb"))

        print("processing reference data")
        ref_collection = db["citations"]
        ref_data = list(ref_collection.find())
        for ref in ref_data:
            id = ref["id"]
            if id in papers_dict:
                if "references" not in papers_dict[id]:
                    papers_dict[id]["references"] = [ref["cited"]]
                else:
                    papers_dict[id]["references"].append(ref["cited"])

        pickle.dump(papers_dict, open("papers_dict.p", "wb"))

    print("loading pickle files")
    papers_dict = pickle.load(open("papers_dict.p", "rb"))
    author_dict = pickle.load(open("author_dict.p", "rb"))
    ground_truth = pickle.load(open("ground_truth.p", "rb"))

    print("generating test data")
    test_data = []
    for author in ground_truth:
        print(f"adding author {author} to test data")
        test_data = test_data + list(set(author_dict[author]) - set(test_data))

    print("generating recommendation system")
    system = Recommender(papers_dict)

    # Initialize lists to store evaluation results
    precision_scores = []
    recall_scores = []
    map_scores = []

    print("Beginning test iterations")
    for x in range(100):
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
            recommendations = system.recommend(poid)
            rec_ids = [paper["id"] for paper in recommendations]

            true_positives = len(set(actual).intersection(rec_ids))
            precision = true_positives / len(recommendations) if len(recommendations) > 0 else 0
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

