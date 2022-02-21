from pathlib import Path
from unittest.mock import patch

import numpy as np

from multi2convai.data.vocab import LabelVocab, WordVocab
from multi2convai.models.logistic_regression import LogisticRegression
from multi2convai.pipelines.base import BasePipelineConfig
from multi2convai.pipelines.inference.base import ClassificationConfig
from multi2convai.pipelines.inference.logistic_regression_base import (
    LogisticRegressionBasePipeline,
    LogisticRegressionConfig,
)
from multi2convai.pipelines.inference.logistic_regression_fasttext import (
    FasttextConfig,
    LogisticRegressionFasttextConfig,
    LogisticRegressionFasttextPipeline,
)
from multi2convai.preprocessing.embedding import Embedding
from multi2convai.preprocessing.preprocessor import Preprocessor
from multi2convai.preprocessing.word_tokenizer import WordTokenizer


class TestFasttextConfig:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given
        embedding_path = Path("embedding_path")
        vocabulary_path = Path("vocabulary_path")

        # When
        test_result = FasttextConfig(
            embedding_path,
            vocabulary_path,
        )

        # Then
        assert test_result is not None
        assert isinstance(test_result, FasttextConfig)
        assert isinstance(test_result, BasePipelineConfig)


class TestLogisticRegressionFasttextConfig:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given
        model_file = Path("model_file")
        embedding_path = Path("embedding_path")
        vocabulary_path = Path("vocabulary_path")

        # When
        test_result = LogisticRegressionFasttextConfig(
            model_file,
            embedding_path,
            vocabulary_path,
        )

        # Then
        assert test_result is not None
        assert isinstance(test_result, LogisticRegressionFasttextConfig)
        assert isinstance(test_result, LogisticRegressionConfig)
        assert isinstance(test_result, FasttextConfig)
        assert test_result.model_file == model_file
        assert test_result.embedding_path == embedding_path
        assert test_result.vocabulary_path == vocabulary_path


class TestLogisticRegressionPipeline:
    def setup(self):
        pass

    def setup_class(self):
        self.model_config = LogisticRegressionFasttextConfig(
            Path("model_file"),
            Path("embedding_path"),
            Path("vocab_path"),
        )
        self.config = ClassificationConfig(
            "en", "domain", Path("label/dict/file"), self.model_config
        )

    def test_create_object(self):
        # Given

        # When
        test_result = LogisticRegressionFasttextPipeline(self.config)

        # Then
        assert test_result is not None
        assert isinstance(test_result, LogisticRegressionFasttextPipeline)
        assert isinstance(test_result, LogisticRegressionBasePipeline)
        assert test_result.config == self.config

    @patch(
        "multi2convai.pipelines.inference.logistic_regression_fasttext.WordVocabLoader"
    )
    @patch(
        "multi2convai.pipelines.inference.logistic_regression_fasttext.EmbeddingLoader"
    )
    @patch("multi2convai.pipelines.inference.logistic_regression_base.DictLoader")
    @patch(
        "multi2convai.pipelines.inference.logistic_regression_base.LogisticRegression"
    )
    @patch("torch.load")
    def test_setup(
        self,
        mock_load,
        mock_model,
        mock_dict_loader,
        mock_embeddings_loader,
        mock_vocab_loader,
    ):
        # Given
        dim = 4
        labels = {"label_0": 0, "label_1": 1, "label_2": 2}
        mock_dict_loader.load.return_value = labels
        word_vocab = WordVocab.from_list(["this", "is", "a", "test"])
        mock_vocab_loader.load.return_value = word_vocab
        mock_embeddings = np.ones([len(word_vocab), dim])
        mock_embeddings_loader.load.return_value = Embedding("test", mock_embeddings)

        pipeline = LogisticRegressionFasttextPipeline(self.config)

        # When
        pipeline.setup()

        # Then
        assert isinstance(pipeline.preprocessor, Preprocessor)
        assert isinstance(pipeline.vectorizer, Embedding)
        assert len(pipeline.vectorizer) == len(word_vocab)
        assert pipeline.vectorizer.dim == dim
        assert isinstance(pipeline.tokenizer, WordTokenizer)
        assert pipeline.tokenizer.vocab == word_vocab
        assert pipeline.label_vocab == LabelVocab(labels)
        assert pipeline.model is not None
        assert mock_load.call_count == 1
        assert mock_model.call_count == 1
        mock_model.assert_called_with(input_dim=dim, output_dim=len(labels))

    def test_get_metadata(self):
        # Given
        dim = 4
        labels = {"label_0": 0, "label_1": 1, "label_2": 2}
        word_vocab = WordVocab.from_list(["this", "is", "a", "test"])
        vector = np.ones([len(word_vocab), dim])

        pipeline = LogisticRegressionFasttextPipeline(self.config)

        pipeline.preprocessor = Preprocessor()
        pipeline.vectorizer = Embedding("fasttext", vector)
        pipeline.label_vocab = LabelVocab(labels)
        pipeline.tokenizer = WordTokenizer(word_vocab)
        pipeline.model = LogisticRegression(4, 2)
        pipeline.startup_time = 123.456

        # When
        test_meta_dict = pipeline.get_metadata()

        # Then
        assert test_meta_dict is not None
        assert None not in test_meta_dict.values()

        assert test_meta_dict["language"] == self.config.language
        assert test_meta_dict["domain"] == self.config.domain
        assert test_meta_dict["pipeline_type"] == "LogisticRegressionFasttextPipeline"
        assert test_meta_dict["startup_time"] == 123.456
        assert test_meta_dict["model_type"] == "LogisticRegression"
        assert test_meta_dict["model_supported_classes"] == list(labels.keys())
        assert test_meta_dict["embedding_type"] == "fasttext"
        assert test_meta_dict["embedding_size"] == dim
