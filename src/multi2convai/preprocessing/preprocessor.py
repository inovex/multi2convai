import re
import string
from abc import ABC, abstractmethod
from typing import List


class PreprocessorStep(ABC):
    """Abstract base class for preprocessing steps."""

    @abstractmethod
    def execute(self, text: str) -> str:
        raise NotImplementedError


class ReplaceWhitespaces(PreprocessorStep):
    """Prepressing step which replaces multiple subsequent whitespaces by a single one."""

    def execute(self, text: str) -> str:
        return re.sub(r"\s+", " ", text)


class ReplaceNumerics(PreprocessorStep):
    """Preprocessing step which replaces digits in a given string by zeros."""

    def execute(self, text: str) -> str:
        return re.sub(r"\d+", "0", text)


class Lowercase(PreprocessorStep):
    """Preprocessing step which lowercases the given text."""

    def execute(self, text: str) -> str:
        return text.lower()


class RemovePunctuation(PreprocessorStep):
    """Preprocessing step which removes punctuation from a given text."""

    def __init__(self):
        self.translator = str.maketrans("", "", string.punctuation)

    def execute(self, text: str) -> str:
        text = text.translate(self.translator)
        text = text.replace("=", " ")
        text = text.replace("%", " ")

        return text


class Preprocessor:
    """Preprocessor applies a series of standard string manipulations on a given list of texts. Per default, the
    preprocessor replaces multiple subsequent whitespaces, lowercases, replaces digits with zeros and removes
    punctuation from the given texts.

    Args:
        lower (bool): will lowercase given texts during preprocessing if True.
        replace_digits (bool): will replace digits with zero during preprocessing if True.
        remove_punctuation (bool): will remove punctuations during preprocessing if True.
    """

    def __init__(
        self,
        lower: bool = True,
        replace_digits: bool = True,
        remove_punctuation: bool = True,
    ):
        self.steps = [ReplaceWhitespaces()]

        if lower:
            self.steps.append(Lowercase())

        if replace_digits:
            self.steps.append(ReplaceNumerics())

        if remove_punctuation:
            self.steps.append(RemovePunctuation())

    def preprocess(self, texts: List[str]) -> List[str]:
        """Applies preprocessing to a given list of text.

        Args:
            texts (List[str]): list of text to be preprocessed

        Returns:
            List[str]: preprocessed texts
        """
        result = []

        for text in texts:
            for step in self.steps:
                text = step.execute(text)
            result.append(text)

        return result
