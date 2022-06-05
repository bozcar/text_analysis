import glob
from pathlib import Path

from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd


def create_tokens(data_dir: str) -> pd.DataFrame:
    data_path = Path(data_dir)

    file_docs: list[list[str]] = []

    datafiles = [Path(n) for n in glob.glob(str(data_path / "*.txt"))]

    for file in datafiles:
        with open(file, 'r') as f:
            tokens = sent_tokenize(f.read())

            for sentence in tokens:
                data = [sentence.lower(), file.stem]
                file_docs.append(data)

    output = pd.DataFrame(
        np.array(file_docs),
        columns=['DESCRIPTION', 'TITLE']
    )
    return output
