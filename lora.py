from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  
from .half_precision import CustomLinear


class CustomLoRALinear(CustomLinear):
    custom_lora_a: torch.nn.Module
    custom_lora_b: torch.nn.Module

    def __init__(
        self,
        input_features: int,
        output_features: int,
        custom_lora_dim: int,
        use_bias: bool = True,
    ) -> None:

        super().__init__(input_features, output_features, use_bias) 
        self.custom_lora_a = torch.nn.Linear(input_features, custom_lora_dim, bias=False, dtype=torch.float32) # lora layer
        self.custom_lora_b = torch.nn.Linear(custom_lora_dim, output_features, bias=False, dtype=torch.float32) # lora layer
 
        torch.nn.init.kaiming_uniform_(self.custom_lora_a.weight)
        torch.nn.init.zeros_(self.custom_lora_b.weight)

        self.custom_lora_a.weight.requires_grad = True # allow Lora to be trainable
        self.custom_lora_b.weight.requires_grad = True # allow lora to be trainable
        self.weight.requires_grad = False # set CustomLinear as NOT trainable
        self.bias.requires_grad = False # set CustomLinear as not trainable

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor = input_tensor.to(self.weight.dtype) # input x
        output = super().forward(input_tensor) # CustomLinear output
        custom_lora_output = self.custom_lora_b(self.custom_lora_a(input_tensor.to(torch.float32))) # lora output
        combined_output = output.to(torch.float32) + custom_lora_output # add the two, ensure CustomLinear output is torch 32 to match lora
        return combined_output.to(input_tensor.dtype) # cast output back to original input_tensor dtype



class CustomLoraBigNet(torch.nn.Module):
    class CustomBlock(torch.nn.Module):
        def __init__(self, channels: int, custom_lora_dim: int):
            super().__init__()
            self.model = torch.nn.Sequential(
              CustomLoRALinear(channels, channels, custom_lora_dim),
              torch.nn.ReLU(),
              CustomLoRALinear(channels, channels, custom_lora_dim),
              torch.nn.ReLU(),
              CustomLoRALinear(channels, channels, custom_lora_dim),
            )


        def forward(self, input_tensor: torch.Tensor):
            return self.model(input_tensor) + input_tensor

    def __init__(self, custom_lora_dim: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.CustomBlock(BIGNET_DIM, custom_lora_dim),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim),
            LayerNorm(BIGNET_DIM),
            self.CustomBlock(BIGNET_DIM, custom_lora_dim),
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return self.model(input_tensor)


def load_custom_lora(path: Path | None) -> CustomLoraBigNet:
    net = CustomLoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
