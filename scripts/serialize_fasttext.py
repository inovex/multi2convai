#!/usr/bin/env python
import logging
import pickle
import sys
from typing import Dict, Tuple

import click
import numpy as np

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-r",
    "--raw-path",
    "raw_path",
    required=True,
    type=str,
    help="Path to file of raw fasttext embeddings",
)
@click.option(
    "-v",
    "--vocab-path",
    "vocab_path",
    required=True,
    type=str,
    help="Path to file in which vocab should be written",
)
@click.option(
    "-e",
    "--embeddings-path",
    "embeddings_path",
    required=True,
    type=str,
    help="Path to file in which embedding vectors should be written",
)
@click.option(
    "-n",
    "--top-n",
    "top_n",
    required=False,
    type=int,
    help="Number of vocabulary words to serialize (default is whole vocabulary)",
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING, default=True)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
def main(raw_path: str, vocab_path: str, embeddings_path: str, top_n: int, log_level):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    vocab, embeddings = load_fasttext_raw(raw_path, top_n)
    serialize_vocab(vocab_path, vocab)
    serialize_embeddings(embeddings_path, embeddings)


def load_fasttext_raw(
    filename: str, top_n: int = -1, skip: bool = True
) -> Tuple[Dict[str, int], np.array]:
    """Loads vocab and embeddings from given fasttext file.
    Slighlty adapted from https://fasttext.cc/docs/en/english-vectors.html.

    Args:
            filename (str): name of raw fasttext embeddings
            top_n (int): restricts vocabulary to top n tokens
            skip (bool): ignores first line if true

    Returns:
            vocab (Dict[str, int]): mapping from token to vocab indices
            embeddings (np.array): word embeddings of shape vocab size x vector dimension
    """
    vocab = {}
    embeddings = []

    logging.info(
        f"Start reading {top_n if top_n > 0 else ''} raw embeddings from '{filename}'"
    )
    with open(filename) as fin:
        if skip:
            next(fin)
        for i, line in enumerate(fin):
            if i < top_n or top_n == -1:
                word, vec = line.rstrip().split(" ", 1)
                vocab[word] = len(vocab)
                embeddings.append(np.array(vec.split(), dtype=np.float32))
            else:
                break

    assert len(vocab) == len(
        embeddings
    ), f"Expected matching lengths of vocab and vectors, but got {len(vocab)} and {len(embeddings)}"
    logging.info(f"Loaded {len(embeddings)} embeddings successfully")

    embeddings = np.array(embeddings, dtype=np.float32)
    return vocab, embeddings


def serialize_vocab(vocab_path: str, vocab: Dict[str, int]) -> None:
    """Writes the vocab to disk.

    Args:
            vocab_path (str): path to filename in which vocab should be written
            vocab (Dict[str, int]): mapping from token to vocab indices
    """
    logging.info(f"Write vocab to '{vocab_path}'")
    pickle.dump(vocab, open(vocab_path, "wb+"))
    logging.info("Stored vocab successfully")


def serialize_embeddings(embeddings_path: str, embeddings: np.array) -> None:
    """Writes the embeddings to disk.

    Args:
            embeddings_path (str): path to filename in which embeddings should be written
            embeddings (np.array): word embeddings of shape vocab size x vector dimension
    """
    logging.info(f"Write embeddings to '{embeddings_path}'")
    np.save(open(embeddings_path, "wb+"), embeddings)
    logging.info("Stored embeddings successfully")


if __name__ == "__main__":
    main()
