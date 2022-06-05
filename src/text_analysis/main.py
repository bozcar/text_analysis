import pandas as pd

from sentenceid import create_tokens
from comparison import tf_idf


def main():
    TEXT_PATH = "./texts/" # The directory where the texts are saved

    tokens = create_tokens(TEXT_PATH)

    token_df = pd.DataFrame.from_dict(
        tokens,
        orient='index',
        columns=['DESCRIPTION']
    )

    tf_idf(token_df)


if __name__ == '__main__':
    main()
