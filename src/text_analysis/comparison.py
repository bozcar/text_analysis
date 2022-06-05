from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tf_idf(index_sentance_data):

    output = defaultdict(list)

    df = pd.DataFrame(columns=["ID","DESCRIPTION"], data=np.matrix(index_sentance_data))

    corpus = list(df["DESCRIPTION"].values)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    threshold = 0.4

    # This is inefficient, it will do everything twice
    for x in range(0,X.shape[0]):
        for y in range(x,X.shape[0]):
            if(x!=y):
                similarity = cosine_similarity(X[x],X[y])
                if(similarity>threshold):
                    output[x].append((y, similarity))
                    print(df["ID"][x],":",corpus[x])
                    print(df["ID"][y],":",corpus[y])
                    print("Cosine similarity:",similarity)
                    print()
    return output