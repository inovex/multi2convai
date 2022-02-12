from abc import ABC, abstractmethod
from typing import Any


class BasePipelineConfig:
    """Base config."""

    def __init__(self, *args, **kwargs):
        pass


class BasePipeline(ABC):
    """Abstract base class for end-to-end pipelines.

    Args:
        config (:class:`~BasePipelineConfig`): pipeline configuration
    """

    def __init__(self, config: BasePipelineConfig):
        self.config = config

    @abstractmethod
    def setup(self) -> None:
        """Performs initial setups of the pipeline. Intended to contain all heavy operations that need to be executed
        once while setting up, e.g. loading models into memory.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Runs pipeline end-to-end. Child class needs to overwrite with specific interfaces for input and output."""
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> dict:
        """Receives the currently loaded model as metadata."""
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """Resets parameters assigned during runtime."""
        raise NotImplementedError
