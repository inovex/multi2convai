import pytest

from multi2convai.data.vocab import WordVocab
from multi2convai.preprocessing.word_tokenizer import WordTokenizer


class TestWordTokenizer:
    def setup(self):
        pass

    def setup_class(self):
        self.vocab = WordVocab.from_list(
            ["this", "is", "a", "test", "second", "example", "repetition", "of", "the"]
        )

    def test_create_object(self):
        # Given

        # When
        test_result = WordTokenizer(self.vocab)

        # Then
        assert test_result is not None
        assert isinstance(test_result, WordTokenizer)
        assert test_result.vocab == self.vocab

    def test_split(self):
        # Given
        tokenizer = WordTokenizer()
        texts = ["this is a test", "second example"]
        split_texts = [["this", "is", "a", "test"], ["second", "example"]]

        # When
        test_result = tokenizer.split(texts)

        # Then
        assert test_result is not None
        assert test_result == split_texts

    def test_encode(self):
        # Given
        tokenizer = WordTokenizer(self.vocab)
        split_texts = [["this", "is", "a", "test"], ["second", "example"]]
        encodings = [[0, 1, 2, 3], [4, 5]]

        # When
        test_result = tokenizer.encode(split_texts)

        # Then
        assert test_result is not None
        assert test_result == encodings

    def test_encode_raises_value_error_if_no_vocab_provided(self):
        tokenizer = WordTokenizer()
        split_texts = [["this", "is", "a", "test"], ["second", "example"]]

        with pytest.raises(
            ValueError, match="Please provide vocab to use encode method."
        ):
            _ = tokenizer.encode(split_texts)

    @pytest.mark.parametrize(
        ["texts", "expected_values"],
        [
            [
                ["this is a test"],
                {"input_ids": [[0, 1, 2, 3]]},
            ],
            [
                [
                    "this is a test",
                    "second example",
                    "this is a repetition of the second example",
                ],
                {"input_ids": [[0, 1, 2, 3], [4, 5], [0, 1, 2, 6, 7, 8, 4, 5]]},
            ],
        ],
    )
    def test_tokenize(self, texts, expected_values):
        # Given
        tokenizer = WordTokenizer(self.vocab)

        # When
        test_result = tokenizer.tokenize(texts)

        # Then
        assert test_result is not None
        assert test_result == expected_values
