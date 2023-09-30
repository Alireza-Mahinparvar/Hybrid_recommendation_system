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

def user_studies(recommended_papers):
    print("User Studies of Recommended Papers:")

    user_feedback = []

    for i in range(len(recommended_papers)):
        paper = recommended_papers[i]
        pprint.pprint(f"Rank {i + 1} - Paper ID: {paper['id']}")
        pprint.pprint(f"Title: {paper['paper title']}")
        pprint.pprint(f"Abstract: {paper.get('abstract', 'None')}")

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

    paper_id = '322'
    paper_of_interest = papers_dict[paper_id]
    recommendations = system.recommend(paper_id)

    print("Paper of Interest:")
    pprint.pprint(f"Title: {paper_of_interest['paper title']}")
    pprint.pprint(f"Abstract: {paper_of_interest.get('abstract', 'None')}")

    user_studies(recommendations)
