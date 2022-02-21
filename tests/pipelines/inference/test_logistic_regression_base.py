from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from multi2convai.data.label import Label
from multi2convai.data.vocab import LabelVocab, WordVocab
from multi2convai.pipelines.base import BasePipelineConfig
from multi2convai.pipelines.inference.base import (
    ClassificationConfig,
    ClassificationPipeline,
)
from multi2convai.pipelines.inference.logistic_regression_base import (
    LogisticRegressionBasePipeline,
    LogisticRegressionConfig,
)
from multi2convai.preprocessing.preprocessor import Preprocessor
from multi2convai.preprocessing.word_tokenizer import WordTokenizer


class TestLogisticRegressionConfig:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given
        model_file = Path("model_file")

        # When
        test_result = LogisticRegressionConfig(
            model_file,
        )

        # Then
        assert test_result is not None
        assert isinstance(test_result, LogisticRegressionConfig)
        assert isinstance(test_result, BasePipelineConfig)
        assert test_result.model_file == model_file


class DummyLogisticRegressionBasePipeline(LogisticRegressionBasePipeline):
    def _create_tokenizer(self) -> str:
        return "tokenizer"

    def _create_vectorizer(self) -> str:
        return "vectorizer"


class TestLogisticRegressionBasePipeline:
    def setup(self):
        pass

    def setup_class(self):
        self.model_config = LogisticRegressionConfig(Path("model_file"))
        self.config = ClassificationConfig(
            "language", "domain", Path("label/path"), self.model_config
        )
        self.labels = {"label_0": 0, "label_1": 1, "label_2": 2}

    def test_create_object(self):
        # Given

        # When
        test_result = DummyLogisticRegressionBasePipeline(self.config)

        # Then
        assert test_result is not None
        assert isinstance(test_result, LogisticRegressionBasePipeline)
        assert isinstance(test_result, ClassificationPipeline)
        assert test_result.config == self.config

    def test_create_vectorizer_raises_error(self):
        # Given
        pipeline = DummyLogisticRegressionBasePipeline(self.config)

        # When
        test_result = pipeline._create_vectorizer()

        # Then
        assert test_result is not None
        assert test_result == "vectorizer"

    def test_create_tokenizer_raises_error(self):
        # Given
        pipeline = DummyLogisticRegressionBasePipeline(self.config)

        # When
        test_result = pipeline._create_tokenizer()

        # Then
        assert test_result is not None
        assert test_result == "tokenizer"

    @patch("multi2convai.pipelines.inference.logistic_regression_base.DictLoader")
    @patch(
        "multi2convai.pipelines.inference.logistic_regression_base.LogisticRegression"
    )
    @patch("torch.load")
    def test_setup(self, mock_load, mock_model, mock_dict_loader):
        # Given
        dim = 4
        mock_dict_loader.load.return_value = self.labels

        pipeline = DummyLogisticRegressionBasePipeline(self.config)
        mock_embedding = MagicMock()
        mock_embedding.dim = 4
        pipeline._create_vectorizer = MagicMock(return_value=mock_embedding)

        # When
        pipeline.setup()

        # Then
        assert isinstance(pipeline.preprocessor, Preprocessor)
        assert pipeline.vectorizer == mock_embedding
        assert pipeline.tokenizer == "tokenizer"
        assert pipeline.label_vocab == LabelVocab(self.labels)
        assert pipeline.model is not None
        assert mock_load.call_count == 1
        assert mock_model.call_count == 1
        mock_model.assert_called_with(input_dim=dim, output_dim=len(self.labels))

    def test_run(self):
        # Given
        text = "some input text 1"
        word_vocab = WordVocab.from_list(["input", "0", "text"])
        expected_label = "label_0"
        embedding = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
        outputs = torch.Tensor([[0.9, 0.2]])
        expected_ratio = 0.6682

        pipeline = DummyLogisticRegressionBasePipeline(self.config)
        pipeline.preprocessor = Preprocessor()
        pipeline.vectorizer = MagicMock()
        pipeline.vectorizer.make_vector.return_value = embedding
        pipeline.tokenizer = WordTokenizer(word_vocab)
        pipeline.label_vocab = LabelVocab(self.labels)
        pipeline.model = MagicMock()
        pipeline.model.return_value = outputs

        # When
        test_result = pipeline.run(text)

        # Then
        assert test_result is not None
        assert isinstance(test_result, Label)
        pipeline.vectorizer.make_vector.assert_called_with(
            {"input_ids": [[3, 0, 2, 1]]}
        )
        assert pipeline.model.call_count == 1
        assert test_result.string == expected_label
        assert round(test_result.ratio, 4) == expected_ratio

    def test_get_metadata(self):
        # Given
        pipeline = DummyLogisticRegressionBasePipeline(self.config)
        pipeline.label_vocab = LabelVocab(self.labels)
        vectorizer = MagicMock()
        vectorizer.dim = 20
        vectorizer.type = "fasttext"
        pipeline.vectorizer = vectorizer
        pipeline.startup_time = 123
        pipeline.model = MagicMock()

        # When
        test_result = pipeline.get_metadata()

        # Then
        assert test_result is not None
        assert isinstance(test_result, Dict)
        assert test_result == {
            "language": self.config.language,
            "domain": self.config.domain,
            "pipeline_type": "DummyLogisticRegressionBasePipeline",
            "startup_time": 123,
            "model_type": "MagicMock",
            "model_supported_classes": list(self.labels.keys()),
            "embedding_type": vectorizer.name,
            "embedding_size": vectorizer.dim,
        }
