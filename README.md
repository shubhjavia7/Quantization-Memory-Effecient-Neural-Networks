# Quantization-Memory-Effecient-Neural-Networks
In this repo, I have created four different versions of a simple high-dimensional multi-layer perceptron. These four versions all aim to reduce memory during training by playing with the network's weights and architecture. I have implemented half-linear, low-rank adapter (LoRA), low precision, and quantized LoRA.

**Half-Precision Networks:** Reduced memory usage by 50% through the implementation of custom layers that store weights in float16 while performing computations in float32.
**LoRA Integration:** Designed trainable low-rank adapters for efficient fine-tuning, enabling parameter-efficient updates while freezing the original model weights.
**4-Bit Quantization:** Achieved a 7x reduction in memory usage by implementing custom quantization and dequantization techniques for neural network weights.
**QLoRA:** Combined 4-bit quantization with LoRA to create a highly memory-efficient and fine-tunable model, suitable for deployment in resource-constrained environments.


