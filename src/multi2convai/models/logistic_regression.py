from torch import Tensor
from torch.nn import Linear, Module


class LogisticRegression(Module):
    """Logistic Regression module.

    Args:
        input_dim (int): size of the input dimension.
        output_dim (int): size of the output dimension.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = Linear(input_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Applies the logistic regression to the given input tensor.

        Args:
            x (Tensor): encoded inputs of shape batchsize x embedding dimension.

        Returns:
            Prediction of shape batchsize x number of classes.
        """

        logits = self.linear(x)
        return logits
