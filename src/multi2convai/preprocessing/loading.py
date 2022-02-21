import json
import pickle
from pathlib import Path
from typing import Dict

import numpy as np

from multi2convai.main.data.vocab import WordVocab
from multi2convai.main.preprocessing.embedding import Embedding


class DictLoader:
    @staticmethod
    def load(path: Path) -> Dict:
        """Loads data from a json file.

        Args:
            path (Path): location of file to load.

        Returns:
            Dict: data loaded from json file.
        """

        with open(path, "r") as file:
            data = json.load(file)

        return data

    @staticmethod
    def save(data_dict: Dict, path: Path) -> None:
        """Saves data to a json file.

        Args:
            data_dict (Dict): data that is saved to the filepath
            path (Path): filepath where the data is supposed to be saved
        """
        with open(path, "w") as file:
            json.dump(data_dict, file)


class WordVocabLoader:
    @staticmethod
    def load(path: Path, UNK: str = "<unk>") -> WordVocab:
        """Loads word vocab from a pickle file.

        Args:
            path (Path): location of your vocabulary file to load.

        Returns:
            :class:`~multi2convai.data.vocab.WordVocab`: vocabulary
        """
        vocab: Dict[str, int] = pickle.load(open(path, "rb"))

        return WordVocab(text2id=vocab, UNK=UNK)


class EmbeddingLoader:
    @staticmethod
    def load(
        path: Path, name: str = "fasttext", add_unk_embedding: bool = True
    ) -> Embedding:
        """Loads embedding vectors from file.

        Args:
            path (Path): location of your embedding file to load.
            add_unk_embedding (bool): appends an embedding for unknown tokens initialized with [0.0] x dim.

        Returns:
            :class:`~multi2convai.preprocessing.embedding.Embedding`: word embeddings (internal shape: vocabulary x dim)
        """

        vector = np.load(path)
        dim = vector.shape[1]

        if add_unk_embedding:
            vector = np.append(vector, np.zeros([1, dim]), axis=0)

        return Embedding(name, vector)
