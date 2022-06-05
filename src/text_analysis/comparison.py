from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tf_idf(df: pd.DataFrame, threshold=0.4):
    corpus = list(df["DESCRIPTION"].values)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)

    output = defaultdict(list)
    for x in range(0,X.shape[0]):
        for y in range(x,X.shape[0]):
            if(x!=y):
                similarity = cosine_similarity(X[x],X[y])
                
                if(similarity>threshold):
                    output[x].append((y, similarity))
                    print(f"{x}: {corpus[x]}")
                    print(f"{y}: {corpus[y]}")
                    print(f"Cosine similarity: {similarity}")
                    print()

    return output
