from itertools import combinations

class CollaborativeFilteringModule:
    def __init__(self, refs: dict = {}):
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

    def get_total_score(self, paper1: int, paper2: int) -> float:
        return (self.get_cooccurring_score(paper1, paper2) + self.get_cooccurred_score(paper1, paper2)) / 2

if __name__ == '__main__':
    import json
    import pymongo
    from crawler import Crawler

    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
    db = client["Aminer"]  # Replace with your database name
    collection = db["Aminer_full_data"]  # Replace with your collection name

    # Query the MongoDB collection to retrieve the data
    data = list(collection.find())

    # Create a dictionary to store the papers indexed by their ids
    papers_dict = {paper["id"]: paper for paper in data}

    print("crawling")
    c = Crawler(papers_dict)
    subset, candidates = c.get_subset("289052")

    print("generating matrices")
    mod = CollaborativeFilteringModule(subset)


    print("gettting scores")
    cooccurred = mod.get_cooccurred_score("288366", "288611")
    cooccurring = mod.get_cooccurring_score("1834", "9518")
    print(cooccurred)
    print(cooccurring)

