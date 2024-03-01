import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit._script import script


@script
def ripple_linear_func(
    input: torch.Tensor, out_features: int, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Assuming we flatten everything if there are more than 2 input dimensions
    Input dimension: BS x IN
    Weight dimension: OUT x IN x 2
    Bias dimension: OUT x (IN + 1)
    Output dimension: BS x OUT

    Args:
        input (torch.Tensor): The input tensor of shape BS x IN.
        out_features (int): The number of output features.
        weight (torch.Tensor): The weight tensor of shape OUT x IN x 2.
        bias (torch.Tensor): The bias tensor of shape OUT x (IN + 1).

    Returns:
        torch.Tensor: The output tensor of shape BS x OUT.
    """

    # Register output sizes
    input_size = input.size()
    output_size = list(input_size)
    output_size[-1] = out_features

    # perform operation w1 * sin(w2 * x + b2) + b1
    input_flattened = input.view(-1, input_size[-1])
    super_batch_size = input_flattened.size()[0]

    # Create output block
    output_sin_mid = torch.sum(
        weight[:, :, 0].repeat(super_batch_size, 1)
        * torch.sin(
            weight[:, :, 1].repeat(super_batch_size, 1)
            * input_flattened.repeat_interleave(out_features, dim=0)
            + bias[:, 1:].repeat(super_batch_size, 1)
        ),
        1,
    ).view(super_batch_size, out_features)
    output_flattened = output_sin_mid + bias[:, 0]

    # return unflattened tensor
    return output_flattened.view(output_size)


@script
def batch_ripple_linear_func(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    """
    Applies a batched ripple linear function to the input.

    Args:
        input (torch.Tensor): Input tensor of size BS x I.
        weight (torch.Tensor): Weight tensor of size BS x O x I x 2.
        bias (torch.Tensor): Bias tensor of size BS x O x (I + 1).

    Returns:
        torch.Tensor: Output tensor of size BS x O.

    The function applies a batched ripple linear transformation to the input tensor.
    It computes the output by taking the sine of the weighted sum of the input,
    multiplied by a ripple factor, and adding a bias term.
    """

    # Register output sizes
    input_size = input.size()
    output_size = list(input_size)
    output_size[-1] = weight.size()[1]

    output_dim = weight.size()[1]
    input_flattened = input.view(-1, 1, input.size()[-1])
    super_batch_size = input_flattened.size()[0]
    batch_multiplier = super_batch_size // input.size()[0]

    output_sin_mid = torch.sum(
        weight[:, :, :, 0].repeat_interleave(batch_multiplier, dim=0)
        * torch.sin(
            weight[:, :, :, 1].repeat_interleave(batch_multiplier, dim=0)
            * input_flattened.repeat((1, output_dim, 1))
            + bias[:, :, 1:].repeat_interleave(batch_multiplier, dim=0)
        ),
        2,
    )
    output_with_bias = output_sin_mid + bias[:, :, 0].repeat_interleave(
        batch_multiplier, dim=0
    )
    return output_with_bias.view(output_size)


class RippleLinear(nn.Module):
    """
    A simple trigonometric linear layer composed of trigonometric neurons; experimental
    neuron type to avoid segmenting the classification field to piece-wise linear segments.
    Should work exactly like the regular input layer, but this time we have ~3n parameters with biases
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initializes the RippleNet module.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool, optional): If True, adds a learnable bias to the output. Defaults to True.
            device (str or torch.device, optional): Device on which to allocate the tensors. Defaults to None.
            dtype (torch.dtype, optional): Data type of the weight and bias tensors. Defaults to None.
        """
        super().__init__()

        factory_kwargs = {"device": device, "dtype": dtype}

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features, 2), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((out_features, in_features + 1), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset the parameters of the layer.

        This method initializes the weight and bias parameters of the layer using the Kaiming uniform initialization
        method. The weight parameter is initialized with a Kaiming uniform distribution with a=math.sqrt(5), which is
        equivalent to initializing with uniform(-1/sqrt(in_features), 1/sqrt(in_features)). The bias parameter is
        initialized with a uniform distribution between -bound and bound, where bound is 1 / math.sqrt(fan_in) if
        fan_in > 0, otherwise it is set to 0.

        Returns:
            None
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the RippleNet module.

        Args:
            input (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return batch_ripple_linear_func(
            input,
            self.weight.unsqueeze(0).repeat(input.shape[0]),
            self.bias.unsqueeze(0).repeat(input.shape[0]),
        )

    def extra_repr(self) -> str:
        """
        Returns a string representation of the module's extra configuration.

        Returns:
            str: A string representation of the module's extra configuration.
        """
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
