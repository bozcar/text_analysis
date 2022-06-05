import pandas as pd

import comparison
from sentenceid import create_tokens


def main():
    TEXT_PATH = "./texts/"

    tokens = create_tokens(TEXT_PATH)

    token_df = pd.DataFrame.from_dict(
        tokens,
        orient='index',
        columns=['DESCRIPTION']
    )

    print(token_df)


if __name__ == '__main__':
    main()
