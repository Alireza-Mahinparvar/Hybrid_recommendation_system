from crawler import Crawler
from collaborative_filtering import CollaborativeFilteringModule
from content_based import ContentBasedModule

class Recommender:
    def __init__(self, refs: list):
        self.refs = refs
        self.crawler = Crawler(refs)
        self.content_based = ContentBasedModule()
        self.collab_filter = None

    def _normalize_score(self, content_score: float, collab_score: float) -> float:
        normalized_score = (content_score + collab_score) / 2
        return normalized_score

    def recommend(self, paper: int) -> list:
        print(f"grabbing subset surrounding paper {paper}")
        subset, candidates, new_ids = self.crawler.get_subset(paper)

        print(f"Calculating term frequency for paper {paper}")
        query_tf = self.content_based.term_freq(self.refs[paper])

        print(f"Calculating citation relations for subset")
        self.collab_filter = CollaborativeFilteringModule(subset)

        for candidate in candidates:
            print(f"Calculating content based score for paper {candidate['id']}")
            content_score = self.content_based.cosine_simi(query_tf, self.content_based.term_freq(self.refs[candidate["id"]]))

            print(f"Calculating collaborative filtering score for paper {candidate['id']}")
            collab_score = self.collab_filter.get_total_score(new_ids[paper], new_ids[candidate["id"]])

            candidate['score'] = self._normalize_score(content_score, collab_score)

        candidates = sorted(candidates, key=lambda x: x['score'], reverse=True)
        return candidates[:min(5, len(candidates))]

if __name__ == '__main__':
    import json
    import pprint

    print('opening')
    f = open('aminerv1.json')
    data = json.load(f)
    f.close()
    print('closed')

    system = Recommender(data)

    pprint.pprint(system.recommend(322302))
