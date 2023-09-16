from Baseline1.recommender import Recommender
from HybridModel.basline2 import Baseline2
import pymongo

class HybridModel:
    def __init__(self, data):
        self.papers_dict = {paper["id"]: paper for paper in data}
        self.baseline1 = Recommender(self.papers_dict)
        self.baseline2 = Baseline2(data)

    def recommend(self, paper: str):
        recommendations1 = self.baseline1.recommend(paper)
        recommendations2, ranks = self.baseline2.recommend(paper)

        scores = {}
        for paper in recommendations1:
            if paper["id"] in scores:
                scores[paper["id"]]["score"] = scores[paper["id"]]["score"] + paper["score"]
            else:
                scores[paper["id"]] = paper
        for paper in recommendations2:
            if paper["id"] in scores:
                scores[paper["id"]]["score"] = scores[paper["id"]]["score"] + paper["score"]
            else:
                scores[paper["id"]] = paper

        total = []
        for paper in scores:
            total.append(scores[paper])
        total = sorted(total, key=lambda x: x['score'], reverse=True)

        return total


if __name__ == '__main__':
    import pprint

    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
    db = client["Aminer"]  # Replace with your database name
    collection = db["papers"]  # Replace with your collection name

    # Query the MongoDB collection to retrieve the data
    data = list(collection.find())

    model = HybridModel(data)

    pprint.pprint(model.recommend("322302"))