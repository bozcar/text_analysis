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
