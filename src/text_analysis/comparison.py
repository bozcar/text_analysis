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

                    x_sen, y_sen = df.loc[x, "DESCRIPTION"], df.loc[y, "DESCRIPTION"]
                    x_title, y_title = df.loc[x, "TITLE"], df.loc[y, "TITLE"]

                    print("Match found:\n")

                    print(f"    {x}: {x_sen}\n    from: {x_title}\n")
                    print(f"    {y}: {y_sen}\n    from: {y_title}\n")

                    print(f"    Cosine similarity: {similarity[0][0]:.3f}\n\n")
    return output
