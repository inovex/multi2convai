from torch.nn import Linear, Module

from multi2convai.main.models.logistic_regression import LogisticRegression


class TestLogisticRegression:
    def setup(self):
        pass

    def setup_class(self):
        pass

    def test_create_object(self):
        # Given
        input_dim, output_dim = 4, 2

        # When
        test_result = LogisticRegression(input_dim, output_dim)

        # Then
        assert test_result is not None
        assert isinstance(test_result, LogisticRegression)
        assert isinstance(test_result, Module)
        assert isinstance(test_result.linear, Linear)
        assert test_result.linear.in_features == input_dim
        assert test_result.linear.out_features == output_dim
