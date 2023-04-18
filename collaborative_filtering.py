from itertools import combinations

class CollaborativeFilteringModule:
    def __init__(self, refs: list = []):
        self.refs = refs
        self.cooccurred = self.generate_cooccurred_matrix(refs)
        self.cooccurring = self.generate_cooccurring_matrix(refs)

    def generate_cooccurred_matrix(self, refs: list) -> dict:
        """
        Generate cooccurred matrix for given reference relations
        :param refs: list of dictionaries that contain 'references' key
        :return: cooccurred matrix where matrix[i][j]=1 if they cooccurred
        """
        matrix = {}
        for paper in range(len(refs)):
            if 'references' in refs[paper]:
                # look at each pair of referenced papers, papers cooccurred if they are cited by same paper
                for pair in combinations(refs[paper]['references'], 2):
                    if pair[0] not in matrix:
                        matrix[pair[0]] = {}
                    if pair[1] not in matrix:
                        matrix[pair[1]] = {}
                    print(pair)
                    matrix[pair[0]][pair[1]] = 1
                    matrix[pair[1]][pair[0]] = 1
        return matrix

    def generate_cooccurring_matrix(self, refs: list) -> dict:
        """
        Generate cooccurring matrix for given reference relations
        :param refs: list of dictionaries that contain 'references' key
        :return: cooccurring matrix where matrix[i][j]=1 if they cooccurring
        """
        matrix = {}
        # look at all combinations f papers to see if they cite teh same paper (cooccurring0
        for pair in combinations(list(range(len(refs))), 2):
            if 'references' in refs[pair[0]] and 'references' in refs[pair[1]]:
                if any(i in refs[pair[0]]['references'] for i in refs[pair[1]]['references']):
                    if pair[0] not in matrix:
                        matrix[pair[0]] = {}
                    if pair[1] not in matrix:
                        matrix[pair[1]] = {}
                    print(pair)
                    matrix[pair[0]][pair[1]] = 1
                    matrix[pair[1]][pair[0]] = 1
        return matrix

    def get_cooccurred_score(self, paper1: int, paper2: int) -> float:
        """
        Calculates cooccurred score
        :param paper1: index of paper 1
        :param paper2: index of paper 2
        :return: cooccurred score
        """
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
        total = (j11 + j10)
        return j11 / total if total > 0 else 0

    def get_cooccurring_score(self, paper1: int, paper2: int) -> float:
        """
        Calculates cooccurring score
        :param paper1: index of paper 1
        :param paper2: index of paper 2
        :return: cooccurring score
        """
        row1 = self.cooccurring[paper1]
        row2 = self.cooccurring[paper2]
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

    print('opening')
    f = open('aminerv1.json')
    data = json.load(f)
    f.close()
    print('closed')

    mod = CollaborativeFilteringModule(data)
    cooccurred = mod.get_cooccurred_score(357875, 214023)
    cooccurring = mod.get_cooccurring_score(322302, 17)
    print(cooccurred)
    print(cooccurring)


