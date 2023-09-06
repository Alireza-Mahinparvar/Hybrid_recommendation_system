import re
import math
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

class ContentBasedModule:
    def __init__(self):
        pass

    def term_freq(self, paper: dict) -> dict:
        """
        Calculates term frequency for given paper
        :param paper: dict containing 'title' and/or 'abstract'
        :return: dictionairy containing word frequencies of form
        {
            'word': term_frequency
        }
        """
        f_dict = {}
        count = 0
        fields = ['paper title', 'abstract', 'keywords']
        for field in fields:
            if field in paper:
                string = paper[field]
                # Replace all single characters with a space
                string = re.sub(r'\b[a-zA-Z]\b', ' ', string)
                # Replace all double spaces with one space
                string = re.sub(' +', ' ', string)
                # Remove leading and trailing spaces
                string = string.strip().lower()
                words = list(string.split(" "))
                for word in words:
                    if word not in stop_words:
                        count += 1
                        if word not in f_dict:
                            f_dict[word] = 1
                        elif word in f_dict:
                            f_dict[word] += 1
        for word in f_dict:
            f_dict[word] = f_dict[word] / count
        return f_dict

    def cosine_simi(self, paper1: dict, paper2: dict) -> float:
        """
        Calculate cosine simialrity of 2 word frequency dictionaries
        :param paper1: word frequencies of paper 1
        :param paper2: word frequencies of paper 2
        :return: cosine similarity score
        """
        word_list1 = list(paper1.keys())
        word_list2 = set(paper2.keys())
        dot_prod = 0
        dist1 = 0
        dist2 = 0
        for word in word_list1:
            if word in word_list2:
                dot_prod += paper1[word] * paper2[word]
            dist1 += paper1[word]*paper1[word]
        dist1 = math.sqrt(dist1)
        for word in word_list2:
            dist2 += paper2[word]*paper2[word]
        dist2 = math.sqrt(dist2)
        return dot_prod / (dist1 + dist2)

if __name__ == '__main__':
    import json
    import pprint
    import pymongo

    # Connect to MongoDB
    client = pymongo.MongoClient("mongodb://localhost:27017")  # Update with your MongoDB connection details
    db = client["Aminer"]  # Replace with your database name
    collection = db["papers"]  # Replace with your collection name

    # Query the MongoDB collection to retrieve the data
    data = list(collection.find())

    # Create a dictionary to store the papers indexed by their ids
    papers_dict = {paper["id"]: paper for paper in data}

    mod = ContentBasedModule()
    f1 = mod.term_freq(papers_dict["991585"])
    f2 = mod.term_freq(papers_dict["289052"])
    pprint.pprint(f1)
    pprint.pprint(f2)
    print(mod.cosine_simi(f1, f2))
