from sentenceid import create_tokens
from comparison import tf_idf


def main():
    TEXT_PATH = "./texts/" # The directory where the texts are saved

    tokens = create_tokens(TEXT_PATH)

    tf_idf(
        tokens, 
        threshold=0.4
    )


if __name__ == '__main__':
    main()
