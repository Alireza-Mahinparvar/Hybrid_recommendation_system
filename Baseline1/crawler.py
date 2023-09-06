
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

if __name__ == '__main__':
    import json
    from pprint import pprint
    import pymongo
    from Baseline1.collaborative_filtering import CollaborativeFilteringModule

    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
    db = client["Aminer"]  # Replace with your database name
    collection = db["papers"]  # Replace with your collection name

    # Query the MongoDB collection to retrieve the data
    data = list(collection.find())

    # Create a dictionary to store the papers indexed by their ids
    papers_dict = {paper["id"]: paper for paper in data}

    c = Crawler(papers_dict)
    subset, candidates = c.get_subset("837")
    pprint(subset)
    pprint(candidates)

    # mod = CollaborativeFilteringModule(subset)
    # cooccurred = mod.get_cooccurred_score(new_ids[322302], new_ids[309116])
    # cooccurring = mod.get_cooccurring_score(new_ids[322302], new_ids[351515])
    # print(cooccurred)
    # print(cooccurring)