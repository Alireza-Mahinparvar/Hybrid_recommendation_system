from itertools import combinations

class CollaborativeFilteringModule:
    def __init__(self, refs={}):
        self.refs = refs
        self.cooccurred = self.generate_cooccurred_matrix(refs)

    def generate_cooccurred_matrix(self, refs):
        matrix = {}
        for paper in refs:
            for pair in combinations(refs[paper]['references'], 2):
                print(pair)
                if pair[0] not in matrix:
                    matrix[pair[0]] = {}
                if pair[1] not in matrix:
                    matrix[pair[1]] = {}
                matrix[pair[0]][pair[1]] = 1
                matrix[pair[1]][pair[0]] = 1
        return matrix

    def get_cooccurred_score(self, paper1, paper2):
        row1 = self.cooccurred[paper1]
        row2 = self.cooccurred[paper2]
        j11 = 0
        j10 = 0
        for i in row1:
            r1 = row1.get(i, 0)
            r2 = row2.get(i, 0)
            if r1 == 1 and r2 == 1:
                j11 += 1
            elif r1 == 1 ^ r2 == 1:
                j10 += 1
        return j11 / (j11 + j10)

    def get_cooccurring_score(self, paper1, paper2):
        ref1 = self.refs[paper1]['references']
        ref2 = set(self.refs[paper2]['references'])
        cooccurring = len([ref for ref in ref1 if ref in ref2])
        return cooccurring / (len(ref1) + len(ref2) - cooccurring)

    def get_total_score(self, paper1, paper2):
        return (self.get_cooccurring_score(paper1, paper2) + self.get_cooccurred_score(paper1, paper2)) / 2


