from typing import Dict, List

import numpy as np


class Embedding:
    """Class to embed texts given pretrained word embeddings.

    Args:
        name (str): name of embedding
        vector (np.ndarray): word embeddings (shape: vocab size x dim)
    """

    def __init__(self, name: str, vector: np.ndarray):
        self.name = name
        self.vector = vector

    @property
    def dim(self) -> int:
        """Returns dimensionality of embedding."""
        return self.vector.shape[1]

    def __len__(self) -> int:
        return self.vector.shape[0]

    def make_vector(self, encodings: Dict[str, List[List[int]]]) -> np.ndarray:
        """Takes an encoded list of texts and computes the average text embedding for them.

        Args:
            encodings (Dict[str, List[List[int]]): encoded input texts (shape: batchsize x sequence length)

        Returns:
            np.ndarray: embeddings for the given texts (shape: batchsize x dim)
        """
        batchsize = len(encodings["input_ids"])

        embedding_sum = np.zeros([batchsize, self.dim])
        counter_non_zero_embeddings = np.zeros(batchsize)

        for i, vocab_ids in enumerate(encodings["input_ids"]):
            # extracts word embeddings given encodings and aggregates them
            word_embeddings = self.vector[vocab_ids, :]
            embedding_sum[i] = word_embeddings.sum(axis=0)

            # counts number of non zero embedding in the given word embeddings
            counter_non_zero_embeddings[i] = (word_embeddings.sum(axis=1) != 0).sum()

        counter_non_zero_embeddings = np.maximum(
            counter_non_zero_embeddings, np.ones(batchsize)
        )

        sequence_embeddings = (embedding_sum.T / counter_non_zero_embeddings).T

        return sequence_embeddings
