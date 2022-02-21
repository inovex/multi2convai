import json
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from multi2convai.data.vocab import WordVocab
from multi2convai.preprocessing.embedding import Embedding
from multi2convai.preprocessing.loading import (
    DictLoader,
    EmbeddingLoader,
    WordVocabLoader,
)


class TestDictLoader:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_load(self, tmpdir):
        # Given
        data = {"some_key": "some_value", "another_key": 2, "third_key": True}
        path = tmpdir.join("config.json")
        with open(path, "w") as file:
            json.dump(data, file)

        # When
        test_result = DictLoader.load(Path(path))

        # Then
        assert test_result is not None
        assert test_result == data

    def test_save(self, tmpdir):
        # Given
        data_dict = {"Label1": 0, "Label2": 1, "Label3": 2}
        temp_file = tmpdir.join("tempfile.json")

        # When
        DictLoader.save(data_dict, Path(temp_file))
        with open(temp_file, "r") as file:
            file_data = json.load(file)

        # Then
        assert os.path.isfile(temp_file)
        assert file_data == {"Label1": 0, "Label2": 1, "Label3": 2}


class TestWordVocabLoader:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_load(self, tmpdir):
        # Given
        vocab = {"this": 0, "is": 1, "a": 2, "test": 3}
        vocab_path = tmpdir.join("vocab.pkl")
        expected_result = {"this": 0, "is": 1, "a": 2, "test": 3, "<unk>": 4}

        pickle.dump(vocab, open(vocab_path, "wb"))

        # When
        test_result = WordVocabLoader.load(Path(vocab_path))

        # Then
        assert test_result is not None
        assert isinstance(test_result, WordVocab)
        assert len(test_result) == 5
        assert test_result.to_dict() == expected_result


class TestEmbeddingLoader:
    def setup(self):
        pass

    def setup_class(self):
        pass

    @pytest.mark.parametrize(
        ["add_unk_embedding", "expected_result"],
        [
            [
                False,
                np.array(
                    [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
                ),
            ],
            [
                True,
                np.array(
                    [
                        [0.0, 0.1, 0.2],
                        [1.0, 1.1, 1.2],
                        [2.0, 2.1, 2.2],
                        [3.0, 3.1, 3.2],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            ],
        ],
    )
    def test_load(self, add_unk_embedding: bool, expected_result: np.ndarray, tmpdir):
        # Given
        data = np.array(
            [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
        )
        data_path = tmpdir.join("embedding.npy")

        np.save(str(data_path), data)

        # When
        test_result = EmbeddingLoader.load(
            Path(data_path), "test_embed", add_unk_embedding
        )

        # Then
        assert test_result is not None
        assert isinstance(test_result, Embedding)
        assert test_result.name == "test_embed"
        assert np.array_equal(test_result.vector, expected_result)
        assert len(test_result) == len(expected_result)
        assert test_result.dim == 3
