from pathlib import Path

import pytest

from multi2convai.pipelines.base import BasePipeline, BasePipelineConfig
from multi2convai.pipelines.inference.base import (
    ClassificationConfig,
    ClassificationPipeline,
)
from multi2convai.pipelines.multilingual_domain_mappings import Multi2ConvAIMapping


class DummyPipeline(ClassificationPipeline):
    def setup(self):
        self.dummy_setup = True

    def run(self, text: str) -> str:
        return text

    def get_metadata(self) -> dict:
        return {"language": self.config.language, "domain": self.config.domain}

    def cleanup(self) -> None:
        self.dummy_setup = None


class TestClassificationConfig:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given
        language = "en"
        domain = "domain"
        label_dict_file = Path("some/path")
        model_config = BasePipelineConfig()

        # When
        test_result = ClassificationConfig(
            language, domain, label_dict_file, model_config
        )

        # Then
        assert test_result is not None
        assert isinstance(test_result, BasePipelineConfig)
        assert isinstance(test_result, ClassificationConfig)
        assert test_result.language == language
        assert test_result.domain == domain
        assert test_result.label_dict_file == label_dict_file
        assert test_result.model_config == model_config


class TestClassificationPipeline:
    def setup(self):
        pass

    def setup_class(self):
        self.config = ClassificationConfig(
            "language", "domain", Path("some/path"), BasePipelineConfig()
        )

    def test_create_object(self):
        # Given

        # When
        test_result = DummyPipeline(self.config)

        # Then
        assert test_result is not None
        assert isinstance(test_result, BasePipeline)
        assert isinstance(test_result, ClassificationPipeline)
        assert isinstance(test_result, DummyPipeline)
        assert test_result.config == self.config
        assert test_result.multilingual_domain_mapping == Multi2ConvAIMapping

    def test_setup(self):
        # Given
        pipeline = DummyPipeline(self.config)

        # When
        pipeline.setup()

        # Then
        assert pipeline.dummy_setup is True

    def test_run(self):
        # Given
        pipeline = DummyPipeline(self.config)
        text = "test text"

        # When
        test_result = pipeline.run(text)

        # Then
        assert test_result == text

    def test_get_metadata(self):
        # Given
        pipeline = DummyPipeline(self.config)

        # When
        test_result = pipeline.get_metadata()

        # Then
        assert test_result == {
            "language": self.config.language,
            "domain": self.config.domain,
        }

    def test_cleanup(self):
        # Given
        pipeline = DummyPipeline(self.config)
        pipeline.setup()

        # When
        pipeline.cleanup()

        # Then
        assert pipeline.dummy_setup is None

    @pytest.mark.parametrize(
        ["pipeline_language", "pipeline_domain", "given_language", "expected_result"],
        [
            ("en", "corona", "en", True),
            ("en", "corona", "de", False),
            ("ml", "corona", "it", True),
            ("ml", "corona", "hr", False),
            ("en", "logistik", "en", True),
            ("en", "logistik", "de", False),
            ("ml", "logistik", "hr", True),
            ("ml", "logistik", "it", False),
            ("ml", "quality", "it", True),
            ("ml", "quality", "hr", False),
            ("ml", "student", "en", True),
            ("ml", "student", "hr", False),
        ],
    )
    def test_is_language_supported(
        self,
        pipeline_language: str,
        pipeline_domain: str,
        given_language: str,
        expected_result: bool,
    ):
        # Given
        config = ClassificationConfig(
            pipeline_language, pipeline_domain, Path("some/path"), BasePipelineConfig()
        )
        pipeline = DummyPipeline(config)

        # When
        test_result = pipeline.is_language_supported(given_language)

        # Then
        assert test_result is not None
        assert test_result == expected_result
