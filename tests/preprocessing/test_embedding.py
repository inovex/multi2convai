from typing import List

import numpy as np
import pytest

from multi2convai.preprocessing.embedding import Embedding


class TestEmbedding:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given
        vector = np.ones([10, 3])

        # When
        test_result = Embedding("test_embedding", vector)

        # Then
        assert test_result is not None
        assert isinstance(test_result, Embedding)
        assert np.array_equal(test_result.vector, vector)
        assert test_result.name == "test_embedding"
        assert test_result.dim == 3
        assert len(test_result) == 10

    @pytest.mark.parametrize(
        ["encodings", "expected_values"],
        [
            [  # <unk> is an an example
                [[0, 1, 2, 2, 3]],
                [2.0],
            ],
            [  # <unk> is an example, <unk> text is an example
                [[0, 1, 2, 3], [0, 4, 1, 2, 3]],
                [2.0, 2.5],
            ],
        ],
    )
    def test_make_vector(
        self,
        encodings: List[List[int]],
        expected_values: List[float],
    ):
        # Given
        # 0: <unk>, 1: is, 2: an, 3: example, 4: text
        vocab_size = 5
        dim = 6

        vector = np.linspace([0] * dim, [vocab_size - 1] * dim, vocab_size)
        embedding = Embedding("test_embedding", vector)
        expected_array = np.array([[v] * dim for v in expected_values])

        encoding_dict = {"input_ids": encodings}

        # When
        test_result = embedding.make_vector(encoding_dict)

        # Then
        assert test_result is not None
        assert isinstance(test_result, np.ndarray)
        assert test_result.shape == expected_array.shape
        assert np.array_equal(test_result, expected_array)
