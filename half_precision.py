from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  


class CustomLinear(torch.nn.Linear):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        use_bias: bool = True,
    ) -> None:
        
        super().__init__(input_features, output_features, use_bias)
        self.weight.data = self.weight.data.half() # this will take it down to float 16
        self.bias.data = self.bias.data.half() # this will take bias down to float16
        self.weight.requires_grad = False # set requires_grad to flase.
        self.bias.requires_grad = False
        

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
      
        input_tensor = input_tensor.to(torch.float32) # blow all back up to float32 for the forward pass
        float_weights = self.weight.to(torch.float32)
        float_bias = self.bias.to(torch.float32)
        float_output = torch.nn.functional.linear(input_tensor, float_weights, float_bias)
        return float_output

class CustomBigNet(torch.nn.Module):
    class CustomBlock(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                CustomLinear(channels, channels), # replaced all linear layers with CustomLinear
                torch.nn.ReLU(),
                CustomLinear(channels, channels),
                torch.nn.ReLU(),
                CustomLinear(channels, channels),
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


def load_custom(path: Path | None) -> CustomBigNet:
    net = CustomBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
