from multi2convai.pipelines.base import BasePipeline, BasePipelineConfig


class DummyPipeline(BasePipeline):
    def __init__(self):
        super().__init__(config=None)
        self.dummy_setup: bool = False

    def setup(self):
        self.dummy_setup = True

    def run(self, text: str) -> str:
        return text

    def get_metadata(self) -> dict:
        return {"test": True}

    def cleanup(self) -> None:
        self.dummy_setup = None


class TestBasePipelineConfig:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # When
        test_result = BasePipelineConfig()

        # Then
        assert test_result is not None
        assert isinstance(test_result, BasePipelineConfig)


class TestBasePipeline:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given

        # When
        test_result = DummyPipeline()

        # Then
        assert test_result is not None
        assert isinstance(test_result, BasePipeline)
        assert isinstance(test_result, DummyPipeline)
        assert test_result.dummy_setup is False
        assert test_result.config is None

    def test_setup(self):
        # Given
        pipeline = DummyPipeline()

        # When
        pipeline.setup()

        # Then
        assert pipeline.dummy_setup is True

    def test_run(self):
        # Given
        pipeline = DummyPipeline()
        text = "test text"

        # When
        test_result = pipeline.run(text)

        # Then
        assert test_result == text

    def test_get_metadata(self):
        # Given
        pipeline = DummyPipeline()

        # When
        test_result = pipeline.get_metadata()

        # Then
        assert test_result == {"test": True}

    def test_cleanup(self):
        # Given
        pipeline = DummyPipeline()
        pipeline.setup()

        # When
        pipeline.cleanup()

        # Then
        assert pipeline.dummy_setup is None
