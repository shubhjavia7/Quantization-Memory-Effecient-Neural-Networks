from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import CustomLinear4Bit


class CustomQLoRALinear(CustomLinear4Bit):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        custom_lora_dim: int,
        group_size: int = 16,
        use_bias: bool = True,
    ) -> None:
        super().__init__(input_features, output_features, use_bias, group_size)
        self.requires_grad_(False)

        self.custom_qlora_a = torch.nn.Linear(input_features, custom_lora_dim, bias=False, dtype=torch.float32)
        self.custom_qlora_b = torch.nn.Linear(custom_lora_dim, output_features, bias=False, dtype=torch.float32)

        torch.nn.init.kaiming_uniform_(self.custom_qlora_a.weight)
        torch.nn.init.zeros_(self.custom_qlora_b.weight)

        self.custom_qlora_a.weight.requires_grad = True
        self.custom_qlora_b.weight.requires_grad = True

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:

        input_tensor = input_tensor.to(torch.float32)
        output = super().forward(input_tensor)  # this will have quantized weights
        qlora_output = self.custom_qlora_b(self.custom_qlora_a(input_tensor.to(torch.float32)))
        combined_output = output.to(torch.float32) + qlora_output
        return combined_output.to(torch.float32)


class CustomQLoRABigNet(torch.nn.Module):
    class CustomBlock(torch.nn.Module):
        def __init__(self, channels, custom_lora_dim, group_size):
            super().__init__()

            self.model = torch.nn.Sequential(
              CustomQLoRALinear(channels, channels, custom_lora_dim, group_size),
              torch.nn.ReLU(),
              CustomQLoRALinear(channels, channels, custom_lora_dim, group_size),
              torch.nn.ReLU(),
              CustomQLoRALinear(channels, channels, custom_lora_dim, group_size),
            )


        def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
            return self.model(input_tensor) + input_tensor

    def __init__(self, custom_lora_dim: int = 32, group_size: int = 16):
        super().__init__()

        self.model = torch.nn.Sequential(
            self.CustomBlock(BIGNET_DIM, custom_lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim, group_size),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim, group_size),
        )


    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)


def load_custom_qlora(path: Path | None) -> CustomQLoRABigNet:
    net = CustomQLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
