from typing import List

import pytest

from multi2convai.main.preprocessing.preprocessor import (
    Lowercase,
    Preprocessor,
    PreprocessorStep,
    RemovePunctuation,
    ReplaceNumerics,
    ReplaceWhitespaces,
)


class TestReplaceWhitespaces:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given

        # When
        test_result = ReplaceWhitespaces()

        # Then
        assert test_result is not None
        assert isinstance(test_result, ReplaceWhitespaces)
        assert isinstance(test_result, PreprocessorStep)

    def test_execute(self):
        # Given
        step = ReplaceWhitespaces()
        texts = "This is a    text with whitespaces.\n\nthe end."

        # When
        test_result = step.execute(texts)

        # Then
        assert test_result is not None
        assert test_result == "This is a text with whitespaces. the end."


class TestReplaceNumerics:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given

        # When
        test_result = ReplaceNumerics()

        # Then
        assert test_result is not None
        assert isinstance(test_result, ReplaceNumerics)
        assert isinstance(test_result, PreprocessorStep)

    def test_execute(self):
        # Given
        step = ReplaceNumerics()
        texts = "This is a text with 12345 numbers."

        # When
        test_result = step.execute(texts)

        # Then
        assert test_result is not None
        assert test_result == "This is a text with 0 numbers."


class TestRemovePunctuation:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given

        # When
        test_result = RemovePunctuation()

        # Then
        assert test_result is not None
        assert isinstance(test_result, RemovePunctuation)
        assert isinstance(test_result, PreprocessorStep)

    def test_execute(self):
        # Given
        step = RemovePunctuation()
        texts = "This is a text with no comma, and no period."

        # When
        test_result = step.execute(texts)

        # Then
        assert test_result is not None
        assert test_result == "This is a text with no comma and no period"


class TestLowercase:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given

        # When
        test_result = Lowercase()

        # Then
        assert test_result is not None
        assert isinstance(test_result, Lowercase)
        assert isinstance(test_result, PreprocessorStep)

    def test_execute(self):
        # Given
        step = Lowercase()
        texts = "This is a text in LOWERCASE."

        # When
        test_result = step.execute(texts)

        # Then
        assert test_result is not None
        assert test_result == "this is a text in lowercase."


class TestPreprocessor:
    def setup(self):
        pass

    def setup_class(self):
        pass

    @pytest.mark.parametrize(
        ["lower", "replace_digits", "remove_punctuation", "expected_steps"],
        [
            [
                True,
                True,
                True,
                [ReplaceWhitespaces, Lowercase, ReplaceNumerics, RemovePunctuation],
            ],
            [False, False, False, [ReplaceWhitespaces]],
            [
                False,
                True,
                True,
                [ReplaceWhitespaces, ReplaceNumerics, RemovePunctuation],
            ],
        ],
    )
    def test_create_object(
        self,
        lower: bool,
        replace_digits: bool,
        remove_punctuation: bool,
        expected_steps: List,
    ):
        # Given

        # When
        test_result = Preprocessor(
            lower=lower,
            replace_digits=replace_digits,
            remove_punctuation=remove_punctuation,
        )

        # Then
        assert test_result is not None
        assert isinstance(test_result, Preprocessor)
        assert isinstance(test_result.steps, list)
        assert len(test_result.steps) == len(expected_steps)
        for step, expected_step in zip(test_result.steps, expected_steps):
            assert isinstance(step, expected_step)

    def test_create_object_no_args(self):
        # Given
        expected_steps = [
            ReplaceWhitespaces,
            Lowercase,
            ReplaceNumerics,
            RemovePunctuation,
        ]

        # When
        test_result = Preprocessor()

        # Then
        assert test_result is not None
        assert isinstance(test_result, Preprocessor)
        assert isinstance(test_result.steps, list)
        assert len(test_result.steps) == len(expected_steps)
        for step, expected_step in zip(test_result.steps, expected_steps):
            assert isinstance(step, expected_step)

    @pytest.mark.parametrize(
        ["lower", "replace_digits", "remove_punctuation", "expected_results"],
        [
            [
                False,
                False,
                False,
                [
                    "This is a test with 012 numbers in 3.",
                    "Another text with 33 counts!",
                ],
            ],
            [
                True,
                True,
                True,
                ["this is a test with 0 numbers in 0", "another text with 0 counts"],
            ],
        ],
    )
    def test_preprocess(
        self,
        lower: bool,
        replace_digits: bool,
        remove_punctuation: bool,
        expected_results: List[str],
    ):
        # Given
        texts = [
            "This is a test with 012 numbers in 3.",
            "Another text with\n\n 33 counts!",
        ]
        preprocessor = Preprocessor(
            lower=lower,
            replace_digits=replace_digits,
            remove_punctuation=remove_punctuation,
        )

        # When
        test_results = preprocessor.preprocess(texts)

        # Then
        assert test_results is not None
        assert isinstance(test_results, list)
        assert len(test_results) == len(expected_results)
        for test_result, expected_result in zip(test_results, expected_results):
            assert test_result == expected_result
