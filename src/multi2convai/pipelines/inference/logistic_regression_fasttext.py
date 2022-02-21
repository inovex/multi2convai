from pathlib import Path

from multi2convai.pipelines.base import BasePipelineConfig
from multi2convai.pipelines.inference.logistic_regression_base import (
    LogisticRegressionBasePipeline,
    LogisticRegressionConfig,
)
from multi2convai.preprocessing.embedding import Embedding
from multi2convai.preprocessing.loading import EmbeddingLoader, WordVocabLoader
from multi2convai.preprocessing.word_tokenizer import WordTokenizer


class FasttextConfig(BasePipelineConfig):
    """Configuration for a :class:`~LogisticRegressionFasttextPipeline`.

    Args:
        embedding_path (Path): location of embedding file
        vocabulary_path (Path): location of vocabulary file
    """

    def __init__(self, embedding_path: Path, vocabulary_path: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embedding_path = embedding_path
        self.vocabulary_path = vocabulary_path


class LogisticRegressionFasttextConfig(LogisticRegressionConfig, FasttextConfig):
    """Configuration for a :class:`~LogisticRegressionFasttextPipeline`.

    Args:
        model_file (Path): location of model file
        embedding_path (Path): location of embedding file
        vocabulary_path (Path): location of vocabular file
    """


class LogisticRegressionFasttextPipeline(LogisticRegressionBasePipeline):
    """Pipeline to run a logistic regression model end-to-end."""

    def _create_tokenizer(self) -> WordTokenizer:
        """Defines the tokenizer by the given `vocabulary_path` in the configuration file.

        Returns:
            :class:`~multi2convai.preprocessing.word_tokenizer.WordTokenizer`: tokenizer to turn texts into encodings
        """
        vocab = WordVocabLoader.load(self.config.model_config.vocabulary_path)
        return WordTokenizer(vocab)

    def _create_vectorizer(self) -> Embedding:
        """Defines the vectorizer by the given `embedding_path` in the configuration file.

        Returns:
            :class:`~multi2convai.preprocessing.embedding.Embedding`: vectorizer to turn encodings into embeddings
        """
        return EmbeddingLoader.load(path=self.config.model_config.embedding_path)
