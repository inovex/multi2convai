from multi2convai.main.data.label import Label


class TestLabel:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given
        label_string = "navigate"
        label_ratio = 0.8342

        # When
        test_result = Label(label_string, label_ratio)

        # Then
        assert test_result is not None
        assert isinstance(test_result, Label)
        assert test_result.string == label_string
        assert test_result.ratio == label_ratio

    def test_string_representation(self):
        # Given
        label_string = "navigate"
        label_ratio = 0.8342
        label = Label(label_string, label_ratio)

        # When
        test_result = str(label)

        # Then
        assert test_result is not None
        assert isinstance(test_result, str)
        assert test_result == label_string
