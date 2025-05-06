# Quantization-Memory-Effecient-Neural-Networks
I have created four versions of a simple high-dimensional multi-layer perceptron in this repo. These four versions all aim to reduce memory during training by playing with the network's weights and architecture.

## Bignet
This file defines the original BigNet model, which is a deep neural network with multiple blocks. Each block consists of linear layers followed by ReLU activations and residual connections. The file also includes:
LayerNorm: A custom implementation of layer normalization.
BigNet: The main neural network model.
load function: A utility to load pretrained weights into the BigNet model.

## Half-Precision
A memory-efficient version of BigNet by using half-precision (float16) weights. Key components include:
CustomLinear: A modified linear layer that stores weights and biases in half-precision but performs computations in full precision (float32).
CustomBigNet: A version of BigNet where all linear layers are replaced with CustomLinear.
load_custom: A function to load pretrained weights into CustomBigNet.

## Low Rank adapter (Lora)
A version of BigNet with Low-Rank Adaptation (LoRA), which is a technique for efficient fine-tuning. Key components include:
CustomLoRALinear: A linear layer with LoRA adapters added. The original weights are frozen, and only the LoRA adapters are trainable.
CustomLoraBigNet: A version of BigNet where all linear layers are replaced with CustomLoRALinear.
load_custom_lora: A function to load pretrained weights into CustomLoraBigNet.

## 4-bit quantization (low precision)
A version of BigNet with 4-bit quantization for weights, significantly reducing memory usage. Key components include:
custom_block_quantize: A function to quantize weights into 4-bit precision.
custom_block_dequantize: A function to dequantize 4-bit weights back to full precision for computation.
CustomLinear4Bit: A linear layer that uses 4-bit quantized weights for storage but performs computations in full precision.
CustomBigNet4Bit: A version of BigNet where all linear layers are replaced with CustomLinear4Bit.
load_custom_4bit: A function to load pretrained weights into CustomBigNet4Bit.

## Quantized Lora (Qlora)
A 4-bit quantization and LoRA to create a highly memory-efficient and fine-tunable version of BigNet. Key components include:
CustomQLoRALinear: A linear layer that combines 4-bit quantization with LoRA adapters.
CustomQLoRABigNet: A version of BigNet where all linear layers are replaced with CustomQLoRALinear.
load_custom_qlora: A function to load pretrained weights into CustomQLoRABigNet.



