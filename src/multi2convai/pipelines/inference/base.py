from abc import abstractmethod
from enum import Enum
from pathlib import Path

from multi2convai.data.label import Label
from multi2convai.pipelines.base import BasePipeline, BasePipelineConfig
from multi2convai.pipelines.multilingual_domain_mappings import Multi2ConvAIMapping


class ClassificationConfig(BasePipelineConfig):
    """Configuration for inference classification pipelines.

    Args:
        language (str): language supported by pipeline
        domain (str): domain supported by pipeline
        label_dict_file (Path): path to file storing label mapping
        model_config (:class:`~multi2convai.main.pipelines.base.BasePipelineConfig`): model specifics
    """

    def __init__(
        self,
        language: str,
        domain: str,
        label_dict_file: Path,
        model_config: BasePipelineConfig,
    ):
        super().__init__()
        self.language = language
        self.domain = domain
        self.label_dict_file = label_dict_file
        self.model_config = model_config


class ClassificationPipeline(BasePipeline):
    """Base class for inference classification pipelines.

    Args:
        config (:class:`~ClassificationConfig`): pipeline configuration
        multilingual_domain_mapping (Enum): mapping from domain to language group for multilingual models.
    """

    def __init__(
        self,
        config: ClassificationConfig,
        multilingual_domain_mapping: Enum = Multi2ConvAIMapping,
    ):
        super().__init__(config)
        self.multilingual_domain_mapping = multilingual_domain_mapping

    def is_language_supported(self, language: str) -> bool:
        """Checks whether given language code belongs to one of the supported languages.

        Args:
            language (str): check if this language is supported by the pipeline

        Returns:
            bool: indicating if the given language is supported by the loaded model
        """
        if self.config.language.lower() == "ml":
            # multilingual setting
            supported_languages = self.multilingual_domain_mapping[
                self.config.domain.upper()
            ].value

        else:
            # monolingual setting
            supported_languages = [self.config.language.lower()]

        return language.lower() in supported_languages

    @abstractmethod
    def run(self, text: str, *args, **kwargs) -> Label:
        """Runs a classification on the given text.

        Args:
            text (str): text to be classified

        Returns:
            :class:`~multi2convai.main.data.label.Label`: label with confidence score assigned to the given text
        """
        raise NotImplementedError
