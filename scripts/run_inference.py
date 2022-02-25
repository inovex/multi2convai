#!/usr/bin/env python
import logging
import sys
from pathlib import Path

import click

from multi2convai.pipelines.inference.base import ClassificationConfig
from multi2convai.pipelines.inference.logistic_regression_fasttext import (
    LogisticRegressionFasttextConfig,
    LogisticRegressionFasttextPipeline,
)

_logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "-m",
    "--model-name",
    "model_name",
    required=True,
    type=str,
    help="Name of the model folder to run",
)
@click.option(
    "-p",
    "--path",
    "path",
    required=False,
    type=str,
    default="",
    help="Path to model folder",
)
@click.option("--quiet", "log_level", flag_value=logging.WARNING)
@click.option("-v", "--verbose", "log_level", flag_value=logging.INFO, default=True)
@click.option("-vv", "--very-verbose", "log_level", flag_value=logging.DEBUG)
def main(model_name: str, path: str, log_level):
    logging.basicConfig(
        stream=sys.stdout,
        level=log_level,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    domain = model_name.split("-")[1]
    language = model_name.split("-")[2]

    if path == "":
        # assuming you run script from repository root
        path = Path(f"models/{domain}/{model_name}")
    else:
        path = Path(path)

    model_file = path / "model.pth"
    label_dict_file = path / "label_dict.json"

    embedding_path = Path(
        f"models/embeddings/fasttext/{language}/wiki.200k.{language}.embed"
    )
    vocabulary_path = Path(
        f"models/embeddings/fasttext/{language}/wiki.200k.{language}.vocab"
    )

    model_config = LogisticRegressionFasttextConfig(
        model_file, embedding_path, vocabulary_path
    )
    config = ClassificationConfig(language, domain, label_dict_file, model_config)

    logging.info(f"Create pipeline for config: {model_name}.")
    pipeline = LogisticRegressionFasttextPipeline(config)
    pipeline.setup()
    logging.info(
        f"Created a {pipeline.__class__.__name__} for domain: '{domain}' and language '{language}'."
    )

    alive: bool = True

    while alive:
        text = input("\nEnter your text (type 'stop' to end execution): ")

        if text == "stop":
            alive = False
        else:
            label = pipeline.run(text)
            logging.info(
                f"'{text}' was classified as '{label.string}' (confidence: {round(label.ratio, 4)})"
            )


if __name__ == "__main__":
    main()
