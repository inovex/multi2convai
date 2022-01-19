import pytest

from multi2convai.main.data.vocab import LabelVocab, Vocab, WordVocab


class TestVocab:
    def setup(self):
        pass

    def setup_class(self):
        self.text2id = {"this": 0, "is": 1, "a": 2, "test": 3}
        self.id2text = {v: k for k, v in self.text2id.items()}

    def test_create_object(self):
        # Given

        # When
        test_result = Vocab(self.text2id)

        # Then
        assert test_result is not None
        assert isinstance(test_result, Vocab)
        assert test_result._text2id == self.text2id
        assert test_result._id2text == self.id2text
        assert len(test_result) == 4
        assert test_result.texts == list(self.text2id.keys())
        assert test_result.ids == list(self.id2text.keys())

    def test_from_list(self):
        # Given
        texts = ["this", "is", "a", "test"]

        # When
        test_result = Vocab.from_list(texts)

        # Then
        assert test_result is not None
        assert isinstance(test_result, Vocab)
        assert test_result._text2id == self.text2id
        assert test_result._id2text == self.id2text

    def test_text_to_id(self):
        # Given
        vocab = Vocab(self.text2id)

        # When
        test_result = vocab.text2id("test")

        # Then
        assert test_result is not None
        assert isinstance(test_result, int)
        assert test_result == 3

    def test_text_to_id_with_unknown_text_raises_error(self):
        # Given
        vocab = Vocab(self.text2id)

        # When
        with pytest.raises(NotImplementedError):
            _ = vocab.text2id("unknown")

    def test_texts_to_ids(self):
        # Given
        vocab = Vocab(self.text2id)

        # When
        test_result = vocab.texts2ids(["test", "a", "test", "this", "is"])
        # Then
        assert test_result is not None
        assert isinstance(test_result, list)
        assert len(test_result) == 5
        assert test_result == [3, 2, 3, 0, 1]

    def test_id_to_text(self):
        # Given
        vocab = Vocab(self.text2id)

        # When
        test_result = vocab.id2text(3)

        # Then
        assert test_result is not None
        assert isinstance(test_result, str)
        assert test_result == "test"

    def test_id_to_text_with_unknown_id_raises_error(self):
        # Given
        vocab = Vocab(self.text2id)

        # When & Then
        with pytest.raises(NotImplementedError):
            _ = vocab.id2text(99)

    def test_ids_to_texts(self):
        # Given
        vocab = Vocab(self.text2id)

        # When
        test_result = vocab.ids2texts([3, 2, 3, 0, 1])
        # Then
        assert test_result is not None
        assert isinstance(test_result, list)
        assert len(test_result) == 5
        assert test_result == ["test", "a", "test", "this", "is"]

    def test_to_dict(self):
        # Given
        vocab = Vocab(self.text2id)

        # When
        test_result = vocab.to_dict()

        # Then
        assert test_result is not None
        assert isinstance(test_result, dict)
        assert test_result == self.text2id

    def test_equals(self):
        # Given
        dict_a = {"this": 0, "is": 1, "a": 2, "test": 3}
        dict_b = {"this": 0, "is": 1, "a": 2, "another": 3, "test": 4}
        vocab_1 = Vocab(dict_a)
        vocab_2 = Vocab(dict_a)
        vocab_3 = Vocab(dict_b)

        # When & Then
        assert vocab_1 == vocab_1
        assert vocab_2 == vocab_2
        assert vocab_3 == vocab_3
        assert vocab_1 == vocab_2

        assert vocab_1 != vocab_3
        assert vocab_2 != vocab_3
        assert vocab_1 != dict_a


class TestWordVocab:
    def setup(self):
        self.text2id = {"this": 0, "is": 1, "a": 2, "test": 3}
        self.id2text = {v: k for k, v in self.text2id.items()}

        self.text2id_with_unknown = {"this": 0, "is": 1, "a": 2, "test": 3, "UNK": 4}
        self.id2text_with_unknown = {v: k for k, v in self.text2id_with_unknown.items()}

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given

        # When
        test_result = WordVocab(self.text2id, "UNK")

        # Then
        assert test_result is not None
        assert isinstance(test_result, Vocab)
        assert isinstance(test_result, WordVocab)
        assert test_result._text2id == self.text2id_with_unknown
        assert test_result._id2text == self.id2text_with_unknown
        assert test_result.UNK == "UNK"
        assert len(test_result) == 5
        assert test_result.texts == list(self.text2id_with_unknown.keys())
        assert test_result.ids == list(self.id2text_with_unknown.keys())

    def test_from_list(self):
        # Given
        texts = ["this", "is", "a", "test"]

        # When
        test_result = WordVocab.from_list(texts, "UNK")

        # Then
        assert test_result is not None
        assert isinstance(test_result, Vocab)
        assert test_result._text2id == self.text2id_with_unknown
        assert test_result._id2text == self.id2text_with_unknown

    @pytest.mark.parametrize(["text", "expected_id"], [["test", 3], ["unknown", 4]])
    def test_text_to_id(self, text: str, expected_id: int):
        # Given
        vocab = WordVocab(self.text2id)

        # When
        test_result = vocab.text2id(text)

        # Then
        assert test_result is not None
        assert isinstance(test_result, int)
        assert test_result == expected_id

    def test_texts_to_ids(self):
        # Given
        vocab = WordVocab(self.text2id)

        # When
        test_result = vocab.texts2ids(["test", "a", "test", "this", "is", "understand"])
        # Then
        assert test_result is not None
        assert isinstance(test_result, list)
        assert len(test_result) == 6
        assert test_result == [3, 2, 3, 0, 1, 4]

    def test_id_to_text(self):
        # Given
        vocab = WordVocab(self.text2id)

        # When
        test_result = vocab.id2text(3)

        # Then
        assert test_result is not None
        assert isinstance(test_result, str)
        assert test_result == "test"

    def test_id_to_text_with_unknown_id_raises_error(self):
        # Given
        vocab = WordVocab(self.text2id)

        # When & Then
        with pytest.raises(ValueError) as e:
            _ = vocab.id2text(99)
        assert (
            "Cannot find word with id '99' in WordVocab. Maximum vocab index is '4'."
            == e.value.args[0]
        )

    def test_ids_to_texts(self):
        # Given
        vocab = WordVocab(self.text2id)

        # When
        test_result = vocab.ids2texts([3, 2, 3, 0, 1])
        # Then
        assert test_result is not None
        assert isinstance(test_result, list)
        assert len(test_result) == 5
        assert test_result == ["test", "a", "test", "this", "is"]

    def test_equals(self):
        # Given
        dict_a = {"this": 0, "is": 1, "a": 2, "test": 3}
        dict_b = {"this": 0, "is": 1, "a": 2, "another": 3, "test": 4}
        vocab_1 = WordVocab(dict_a)
        vocab_2 = WordVocab(dict_a)
        vocab_3 = WordVocab(dict_b)
        vocab_4 = Vocab(dict_a)

        # When & Then
        assert vocab_1 == vocab_1
        assert vocab_2 == vocab_2
        assert vocab_3 == vocab_3
        assert vocab_1 == vocab_2

        assert vocab_1 != vocab_3
        assert vocab_2 != vocab_3
        assert vocab_1 != dict_a
        assert vocab_1 != vocab_4


class TestLabelVocab:
    def setup(self):
        pass

    def setup_class(self):
        self.text2id = {"label_0": 0, "label_1": 1, "label_2": 2, "label_3": 3}
        self.id2text = {v: k for k, v in self.text2id.items()}

    def test_create_object(self):
        # Given

        # When
        test_result = LabelVocab(self.text2id)

        # Then
        assert test_result is not None
        assert isinstance(test_result, Vocab)
        assert isinstance(test_result, LabelVocab)
        assert test_result._text2id == self.text2id
        assert test_result._id2text == self.id2text
        assert len(test_result) == 4
        assert test_result.texts == list(self.text2id.keys())
        assert test_result.ids == list(self.id2text.keys())

    def test_from_list(self):
        # Given
        labels = ["label_0", "label_1", "label_2", "label_3"]

        # When
        test_result = LabelVocab.from_list(labels)

        # Then
        assert test_result is not None
        assert isinstance(test_result, Vocab)
        assert test_result._text2id == self.text2id
        assert test_result._id2text == self.id2text

    def test_id_to_text_with_unknown_id_raises_error(self):
        # Given
        vocab = LabelVocab(self.text2id)

        # When & Then
        with pytest.raises(ValueError) as e:
            _ = vocab.id2text(99)
        assert (
            "Cannot find label with id '99' in LabelVocab. Maximum vocab index is '3'."
            == e.value.args[0]
        )

    def test_text_to_id_with_unknown_text_raises_error(self):
        # Given
        vocab = LabelVocab(self.text2id)

        # When & Then
        with pytest.raises(ValueError) as e:
            _ = vocab.text2id("label_unknown")
        assert (
            "Cannot find label with name 'label_unknown' in LabelVocab."
            == e.value.args[0]
        )

    def test_equals(self):
        # Given
        dict_a = {"label_0": 0, "label_1": 1, "label_2": 2}
        dict_b = {"label_0": 0, "label_1": 1, "label_2": 2, "label_3": 3}
        vocab_1 = LabelVocab(dict_a)
        vocab_2 = LabelVocab(dict_a)
        vocab_3 = LabelVocab(dict_b)
        vocab_4 = Vocab(dict_a)

        # When & Then
        assert vocab_1 == vocab_1
        assert vocab_2 == vocab_2
        assert vocab_3 == vocab_3
        assert vocab_1 == vocab_2

        assert vocab_1 != vocab_3
        assert vocab_2 != vocab_3
        assert vocab_1 != dict_a
        assert vocab_1 != vocab_4
