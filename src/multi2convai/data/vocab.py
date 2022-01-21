from abc import abstractmethod
from copy import copy
from typing import Dict, List


class Vocab:
    """Generic vocabulary mapping text to vocab indices and vice versa.

    Args:
        text2id (Dict[str, int]): mapping from strings to vocab indices.
    """

    def __init__(self, text2id: Dict[str, int]):
        self._text2id = text2id
        self._id2text = {value: key for key, value in text2id.items()}

    @classmethod
    def from_list(cls, texts: List[str]):
        """Creates Vocab from list of texts

        Args:
            texts (List[str]): list of texts to create vocabulary from.
        """
        text2id = {text: id for id, text in enumerate(texts)}
        return cls(text2id)

    def text2id(self, text: str) -> int:
        """Returns vocab index of given text

        Args:
            text (str): text to be looked up

        Returns:
            int: vocab index for given text
        """
        if text in self._text2id:
            return self._text2id[text]
        else:
            return self._handle_unknown_text(text)

    def texts2ids(self, texts: List[str]) -> List[int]:
        """Returns vocab ids for a given list of text.

        Args:
            texts (List[str]): texts to be looked up

        Returns:
            List[int]: vocab indices for given texts
        """
        return [self.text2id(text) for text in texts]

    def id2text(self, id: int) -> str:
        """Returns the token of given vocab index

        Args:
            id (int): vocab index

        Returns:
            str: text of given vocab index
        """
        if id in self._id2text.keys():
            return self._id2text[id]
        else:
            return self._handle_unknown_id(id)

    def ids2texts(self, ids: List[int]) -> List[str]:
        """Returns the tokens of given vocab indices

        Args:
            ids (List[int]): vocab indices

        Returns:
            List[str]: texts for given vocab indices
        """
        return [self.id2text(id) for id in ids]

    def to_dict(self) -> Dict[str, int]:
        """Returns vocab as dictionary mapping from texts to ids."""
        return self._text2id

    @property
    def texts(self) -> List[str]:
        """Returns all tokens in vocab in order of their vocab indices."""
        return list(self._text2id.keys())

    @property
    def ids(self) -> List[int]:
        """Returns all vocab indicies."""
        return list(self._id2text.keys())

    @abstractmethod
    def _handle_unknown_text(self, text: str) -> int:
        """Allows subclasses to specify how unknown texts should be handled."""
        raise NotImplementedError

    @abstractmethod
    def _handle_unknown_id(self, id: int) -> str:
        """Allows subclasses to specify how unknown ids should be handled."""
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._text2id)

    def __eq__(self, other: "Vocab") -> bool:
        if type(other) == type(self):
            return self._text2id == other._text2id and self._id2text == other._id2text
        else:
            return False


class WordVocab(Vocab):
    """Word vocabulary mapping words to vocab indices and vice versa."

    Args:
        text2id (Dict[str, int]): mapping from words to vocab indices.
        UNK (str): string used to represent unknown words.
    """

    def __init__(self, text2id: Dict[str, int], UNK: str = "<unk>"):
        # need to copy here to avoid inplace manipulation of original dict
        # causes problems when initializing multiple WordVocabs
        text2id = copy(text2id)
        text2id[UNK] = len(text2id)
        super().__init__(text2id)
        self.UNK = UNK

    @classmethod
    def from_list(cls, texts: List[str], UNK: str = "<unk>"):
        """Creates Vocab from list of texts

        Args:
            texts (List[str]): list of texts to create vocabulary from.
            UNK (str): string used to represent unknown words.
        """
        text2id = {text: id for id, text in enumerate(texts)}
        return cls(text2id, UNK)

    def _handle_unknown_text(self, text: str) -> int:
        """Overwrites parent class and maps unknown texts to the `UNK` vocab index."""
        return self.text2id(self.UNK)

    def _handle_unknown_id(self, id: int):
        """Unknown ids raise a ValueError."""
        raise ValueError(
            f"Cannot find word with id '{id}' in WordVocab. Maximum vocab index is '{len(self)-1}'."
        )


class LabelVocab(Vocab):
    def _handle_unknown_text(self, text: str):
        """Unknown texts raise a ValueError."""
        raise ValueError(f"Cannot find label with name '{text}' in LabelVocab.")

    def _handle_unknown_id(self, id: int):
        """Unknown ids raise a ValueError."""
        raise ValueError(
            f"Cannot find label with id '{id}' in LabelVocab. Maximum vocab index is '{len(self)-1}'."
        )
