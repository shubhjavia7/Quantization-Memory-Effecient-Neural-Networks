from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def custom_block_quantize(input_tensor: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    assert input_tensor.dim() == 1
    assert input_tensor.size(0) % group_size == 0

    input_tensor = input_tensor.view(-1, group_size)
    normalization = input_tensor.abs().max(dim=-1, keepdim=True).values
    normalized_tensor = (input_tensor + normalization) / (2 * normalization)
    quantized_8bit = (normalized_tensor * 15).round().to(torch.int8)
    quantized_4bit = (quantized_8bit[:, ::2] & 0xF) + ((quantized_8bit[:, 1::2] & 0xF) << 4)
    return quantized_4bit, normalization.to(torch.float16)


def custom_block_dequantize(quantized_4bit: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    assert quantized_4bit.dim() == 2

    normalization = normalization.to(torch.float32)
    quantized_8bit = quantized_4bit.new_empty(quantized_4bit.size(0), quantized_4bit.shape[1] * 2)
    quantized_8bit[:, ::2] = quantized_4bit & 0xF
    quantized_8bit[:, 1::2] = (quantized_4bit >> 4) & 0xF
    normalized_tensor = quantized_8bit.to(torch.float32) / 15
    dequantized_tensor = (normalized_tensor * 2 * normalization) - normalization
    return dequantized_tensor.view(-1)


class CustomLinear4Bit(torch.nn.Module):
    def __init__(self, input_features: int, output_features: int, use_bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        self._shape = (output_features, input_features)
        self._group_size = group_size

        self.register_buffer(
            "quantized_weights",
            torch.zeros(output_features * input_features // group_size, group_size // 2, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_normalization",
            torch.zeros(output_features * input_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )
        self._register_load_state_dict_pre_hook(CustomLinear4Bit._load_state_dict_pre_hook, with_module=True)
        # Add in an optional bias
        self.bias = None
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_features, dtype=torch.float32))

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            # Load the original weights and remove them from the state_dict (mark them as loaded)
            original_weights = state_dict[f"{prefix}weight"]  # noqa: F841
            del state_dict[f"{prefix}weight"]
            # Quantize the weights and store them in self.quantized_weights and self.weight_normalization
            flat_weights = original_weights.flatten() 
            self.quantized_weights, self.weight_normalization = custom_block_quantize(flat_weights, self._group_size)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Dequantize and call the layer
            dequantized_weights = custom_block_dequantize(self.quantized_weights, self.weight_normalization)
            output = torch.nn.functional.linear(input_tensor, dequantized_weights.view(self._shape), self.bias)
            return output


class CustomBigNet4Bit(torch.nn.Module):
    """
    A BigNet where all weights are in 4bit precision. Use the CustomLinear4Bit module for this.
     all computation are in float32.
    """
    class CustomBlock(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                CustomLinear4Bit(channels, channels),
                torch.nn.ReLU(),
                CustomLinear4Bit(channels, channels),
                torch.nn.ReLU(),
                CustomLinear4Bit(channels, channels),
            )

        def forward(self, input_tensor):
            return self.model(input_tensor) + input_tensor

    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            self.CustomBlock(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM),
        )

    def forward(self, input_tensor):
        return self.model(input_tensor)


def load_custom_4bit(path: Path | None) -> CustomBigNet4Bit:
    net = CustomBigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net

