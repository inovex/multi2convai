from typing import Dict, List, Optional

from nltk import word_tokenize

from multi2convai.data.vocab import WordVocab


class WordTokenizer:
    """Tokenizer that uses NLTK to split texts and encodes them according to the given Vocab.

    Args:
        vocab (Optional[:class:`~multi2convai.data.vocab.WordVocab`]): word vocab.
    """

    def __init__(self, vocab: Optional[WordVocab] = None):
        self.vocab = vocab

    def split(self, texts: List[str]) -> List[List[str]]:
        """Splits given texts using the NLTK word tokenization.

        Args:
            texts (List[str]): texts to be tokenized

        Returns:
            List[List[str]]: tokens of the input texts
        """
        word_lists = [[word for word in word_tokenize(text)] for text in texts]
        return word_lists

    def encode(self, word_lists: List[List[str]]) -> List[List[int]]:
        """Assigns vocab indices to a list of tokens.

        Args:
            tokenized_texts (List[List[str]]): words to be encoded

        Returns:
            List[List[int]]: vocab indices of given tokens
        """
        if self.vocab:
            encodings = [self.vocab.texts2ids(words) for words in word_lists]
        else:
            raise ValueError("Please provide vocab to use encode method.")

        return encodings

    def _tokenize(self, texts: List[str]) -> List[List[int]]:
        """Tokenizes (splits and encodes) a list of texts.

        Args:
            texts (List[str]): texts to be tokenized

        Returns:
            List[List[int]]: vocab indices of given texts
        """
        splitted_texts = self.split(texts)
        encodings = self.encode(splitted_texts)
        return encodings

    def tokenize(self, texts: List[str]) -> Dict[str, List[List[int]]]:
        """Tokenizes (splits and encodes) a list of texts.

        Args:
            texts (List[str]): list of strings to be tokenized

        Returns:
            Dict[str, List[List[int]]]: tokenized texts
        """
        encodings = {"input_ids": self._tokenize(texts)}

        return encodings
