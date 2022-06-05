import glob
from pathlib import Path

from nltk.tokenize import sent_tokenize
import gensim


def create_tokens(data_dir: str) -> dict:
    data_path = Path(data_dir)

    file_docs: list[str] = []

    datafiles = glob.glob(str(data_path / "*.txt"))

    for file in datafiles:
        with open(file, 'r') as f:
            tokens = sent_tokenize(f.read())

            for sentance in tokens:
                file_docs.append(sentance.lower())

    output = gensim.corpora.Dictionary([file_docs])
    return output













# file_docs = []

# path = '/home/lucy/project/NLTK/WAPOtestsamp'
# for filename in glob.glob(os.path.join(path, '*.txt')):
#    with open(os.path.join(os.getcwd(), filename), 'r') as f:


# #with open ('A412120044 A fight for democrac.txt') as f:
#     tokens = word_tokenize(f.read())
#     for line in tokens:
#         file_docs.append(line)

# #print("Number of documents:",len(file_docs))
# print(tokens)

# with open(os.path.join(os.getcwd(), filename), 'r') as f:
#     tokens = sent_tokenize(f.read())
#     for line in tokens:
#         file_docs.append(line)

# #print(tokens)
# #print("Number of documents:",len(file_docs))

# gen_docs = [[w.lower() for w in sent_tokenize(text)] 
#             for text in file_docs]

# dictionary = gensim.corpora.Dictionary(gen_docs)
# print(dictionary.token2id)



# #corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
# #print(corpus)



# #tf_idf = gensim.models.TfidfModel(corpus)
# #for doc in tf_idf[corpus]:
# #    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])

