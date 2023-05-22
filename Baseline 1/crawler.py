
class Crawler:
    def __init__(self, refs: list = []):
        self.refs = refs

    def get_subset(self, query: int):
        subset = []
        candidates = []
        new_ids = {query: 0}
        subset.append(self.refs[query])

        # get papers citing query and their references
        for i in range(len(self.refs)):
            paper = self.refs[i]
            if 'references' in paper:
                if query in paper['references']:
                    if paper['id'] not in new_ids:
                        new_ids[paper['id']] = len(subset)
                        subset.append(paper)
                    for ref in paper['references']:
                        if ref not in new_ids:
                            new_ids[ref] = len(subset)
                            subset.append(self.refs[ref])
                            candidates.append(self.refs[ref])

        # get query's references and those that cited query's references
        for ref in self.refs[query]['references']:
            if ref not in new_ids:
                new_ids[ref] = len(subset)
                subset.append(self.refs[ref])
                candidates.append(self.refs[ref])
            for i in range(len(self.refs)):
                paper = self.refs[i]
                if 'references' in paper:
                    if ref in paper['references']:
                        if paper['id'] not in new_ids:
                            new_ids[paper['id']] = len(subset)
                            subset.append(paper)
                        if paper not in candidates:
                            candidates.append(paper)

        # change ids for ease
        for paper in subset:
            if 'references' in paper:
                for ref in range(len(paper['references'])):
                    paper['references'][ref] = new_ids.get(paper['references'][ref], paper['references'][ref])

        return subset, candidates, new_ids

if __name__ == '__main__':
    import json
    import pprint
    from collaborative_filtering import CollaborativeFilteringModule

    print('opening')
    f = open('aminerv1.json')
    data = json.load(f)
    f.close()
    print('closed')

    c = Crawler(data)
    subset, candidates, new_ids = c.get_subset(322302)
    print(len(subset))
    print(len(candidates))

    mod = CollaborativeFilteringModule(subset)
    cooccurred = mod.get_cooccurred_score(new_ids[322302], new_ids[309116])
    cooccurring = mod.get_cooccurring_score(new_ids[322302], new_ids[351515])
    print(cooccurred)
    print(cooccurring)