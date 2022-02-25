import logging
import time
from abc import abstractmethod
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from multi2convai.data.label import Label
from multi2convai.data.vocab import LabelVocab
from multi2convai.models.logistic_regression import LogisticRegression
from multi2convai.pipelines.base import BasePipelineConfig
from multi2convai.pipelines.inference.base import ClassificationPipeline
from multi2convai.preprocessing.loading import DictLoader
from multi2convai.preprocessing.preprocessor import Preprocessor


class LogisticRegressionConfig(BasePipelineConfig):
    """Configuration for a logistic regression model.

    Args:
        model_file (Path): location of model file
    """

    def __init__(self, model_file: Path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_file = model_file


class LogisticRegressionBasePipeline(ClassificationPipeline):
    """Base Pipeline to run a logistic regression model end-to-end."""

    def setup(self) -> None:
        """Performs setup:
        - loads embeddings
        - loads label vocab
        - loads tokenizer
        - loads model
        """
        t0 = time.time()
        self.preprocessor = Preprocessor()
        self.vectorizer = self._create_vectorizer()
        self.label_vocab: LabelVocab = self._create_label_vocab()
        self.tokenizer = self._create_tokenizer()
        self.model: LogisticRegression = self._create_model()

        self.startup_time = time.time() - t0

    def run(self, text: str) -> Label:
        """Runs end-to-end pipeline including preprocessing, vectorization and model prediction.

        Args:
            text (str): text to be classified.

        Returns:
            :class:`~multi2convai.data.label.Label`: label assigned to input text.
        """
        with torch.no_grad():
            preprocessed_texts: List[str] = self.preprocessor.preprocess([text])
            encodings: Dict[str, List[List[int]]] = self.tokenizer.tokenize(
                preprocessed_texts
            )

            embeddings: np.ndarray = self.vectorizer.make_vector(encodings)

            inputs: torch.Tensor = torch.Tensor(embeddings)

            outputs: torch.Tensor = self.model(inputs.view(inputs.shape[0], -1))
            outputs: torch.Tensor = F.softmax(outputs, dim=1)

            ratio, predicted = torch.max(outputs.data, 1)
            label = Label(self.label_vocab.id2text(predicted.item()), ratio.item())
            logging.debug(
                "Predict Label: {}, Ratio: {}".format(label.string, label.ratio)
            )

            return label

    def get_metadata(self) -> Dict:
        """Returns details about the pipeline metadata.

        Returns:
            Dict: metadata of the pipeline.
        """
        metadata = {
            "language": self.config.language,
            "domain": self.config.domain,
            "pipeline_type": self.__class__.__name__,
            "startup_time": self.startup_time,
            "model_type": self.model.__class__.__name__,
            "model_supported_classes": self.label_vocab.texts,
            "embedding_type": self.vectorizer.name,
            "embedding_size": self.vectorizer.dim,
        }

        return metadata

    def _create_label_vocab(self) -> LabelVocab:
        """Loads and returns label vocab as specified in config file."""
        labels: Dict[str, int] = DictLoader.load(self.config.label_dict_file)
        return LabelVocab(labels)

    def _create_model(self) -> LogisticRegression:
        """Loads and returns a logistic regression model in evaluation mode."""
        model: LogisticRegression = LogisticRegression(
            input_dim=self.vectorizer.dim, output_dim=len(self.label_vocab)
        )
        model.load_state_dict(torch.load(self.config.model_config.model_file))
        model.eval()

        return model

    @abstractmethod
    def _create_tokenizer(self):
        """Child class needs to implement this method in order to define a specific tokenizer."""
        raise NotImplementedError

    @abstractmethod
    def _create_vectorizer(self):
        """Child class needs to implement this method in order to define a specific vectorizer."""
        raise NotImplementedError

    def cleanup(self) -> None:
        self.preprocessor = None
        self.vectorizer = None
        self.tokenizer = None
        self.label_vocab = None
        self.model = None
        self.startup_time = None
