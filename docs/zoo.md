# Compatibility: the model zoo

Every architecture below runs in the test suite (`pytest -m zoo`; the slowest ones weekly).

<!-- zoo:start -->
61 of 61 architectures wrap, trace to the native model's output (to 1e-4) and make a page — built from random weights or tiny configs, so nothing is downloaded. Regenerate with `DEEPVIS_ZOO_REPORT=1 pytest -m zoo && python docs/zoo_table.py`.

| model | from | framework | layers drawn | parameters | page | |
|---|---|---|---:|---:|---:|---|
| BART (encoder-decoder) | Hugging Face | torch | 138 | 50K | 2.32 MB | ✓ |
| BERT | Hugging Face | torch | 60 | 38K | 1.21 MB | ✓ |
| CLIP (image + text) | Hugging Face | torch | 76 | 29K | 1.54 MB | ✓ |
| DistilBERT | Hugging Face | torch | 50 | 37K | 1.07 MB | ✓ |
| GPT-2 | Hugging Face | torch | 60 | 61K | 0.99 MB | ✓ |
| GPT-NeoX | Hugging Face | torch | 90 | 24K | 1.19 MB | ✓ |
| Gemma | Hugging Face | torch | 81 | 22K | 1.15 MB | ✓ |
| Llama | Hugging Face | torch | 81 | 25K | 1.15 MB | ✓ |
| Mamba (state space) | Hugging Face | torch | 242 | 19K | 1.59 MB | ✓ |
| Mistral | Hugging Face | torch | 85 | 25K | 1.18 MB | ✓ |
| Mixtral (MoE) | Hugging Face | torch | 139 | 62K | 1.4 MB | ✓ |
| Phi | Hugging Face | torch | 92 | 24K | 1.24 MB | ✓ |
| Qwen2 | Hugging Face | torch | 81 | 25K | 1.18 MB | ✓ |
| RoBERTa | Hugging Face | torch | 65 | 38K | 1.25 MB | ✓ |
| Swin | Hugging Face | torch | 92 | 19K | 1.6 MB | ✓ |
| T5 (encoder-decoder) | Hugging Face | torch | 145 | 45K | 2.19 MB | ✓ |
| ViT | Hugging Face | torch | 54 | 24K | 1.28 MB | ✓ |
| Whisper (speech) | Hugging Face | torch | 81 | 31K | 1.6 MB | ✓ |
| wav2vec 2.0 (audio) | Hugging Face | torch | 65 | 27K | 1.73 MB | ✓ |
| Equinox MLP | JAX | equinox | 5 | 435 | 0.37 MB | ✓ |
| Flax linen CNN | JAX | flax | 5 | 5K | 0.34 MB | ✓ |
| Flax nnx MLP | JAX | nnx | 3 | 163 | 0.32 MB | ✓ |
| Keras ConvNeXtTiny | Keras | keras · jax | 156 | 27.8M | 3.59 MB | ✓ |
| Keras EfficientNetB0 | Keras | keras · jax | 239 | 4.0M | 3.5 MB | ✓ |
| Keras MobileNetV3Small | Keras | keras · jax | 181 | 1.5M | 2.53 MB | ✓ |
| Keras ResNet50 | Keras | keras · jax | 176 | 23.6M | 3.66 MB | ✓ |
| Keras Sequential CNN | Keras | keras · jax | 7 | 5K | 0.45 MB | ✓ |
| Keras functional (residual blocks) | Keras | keras · jax | 13 | 3K | 0.51 MB | ✓ |
| Keras text (Bidirectional LSTM, attention) | Keras | keras · jax | 8 | 4K | 0.39 MB | ✓ |
| BERT · dynamo exporter | ONNX | onnx | 44 | 37K | 1.05 MB | ✓ |
| GPT-2 · dynamo exporter | ONNX | onnx | 42 | 64K | 0.86 MB | ✓ |
| LSTM classifier · TorchScript exporter | ONNX | onnx | 4 | 6K | 0.3 MB | ✓ |
| MobileNetV3 · TorchScript exporter | ONNX | onnx | 122 | 2.5M | 2.73 MB | ✓ |
| ResNet-18 · TorchScript exporter | ONNX | onnx | 49 | 11.7M | 1.64 MB | ✓ |
| ResNet-18 · dynamo exporter | ONNX | onnx | 49 | 11.7M | 1.55 MB | ✓ |
| Transformer encoder · dynamo exporter | ONNX | onnx | 60 | 17K | 0.87 MB | ✓ |
| U-Net style (transposed conv, skip) · dynamo exporter | ONNX | onnx | 8 | 2K | 0.4 MB | ✓ |
| ViT · TorchScript exporter | ONNX | onnx | 61 | 24K | 1.26 MB | ✓ |
| VAE (your own, tuple output) | custom | torch | 12 | 340 | 0.42 MB | ✓ |
| convnext_atto | timm | torch | 173 | 3.7M | 3.06 MB | ✓ |
| deit_tiny_patch16_224 | timm | torch | 375 | 5.7M | 5.58 MB | ✓ |
| efficientnet_b0 (timm) | timm | torch | 336 | 5.3M | 7.97 MB | ✓ |
| mixer_s16_224 | timm | torch | 168 | 18.5M | 3.81 MB | ✓ |
| resnet18 (timm) | timm | torch | 85 | 11.7M | 2.03 MB | ✓ |
| vit_tiny_patch16_224 | timm | torch | 375 | 5.7M | 5.58 MB | ✓ |
| convnext_tiny | torchvision | torch | 192 | 28.6M | 3.72 MB | ✓ |
| densenet121 | torchvision | torch | 427 | 8.0M | 8.49 MB | ✓ |
| efficientnet_b0 | torchvision | torch | 256 | 5.3M | 4.66 MB | ✓ |
| fasterrcnn_mobilenet_v3 (detection) | torchvision | torch | 438 | 19.0M | 6.58 MB | ✓ |
| fcn_resnet50 (segmentation) | torchvision | torch | 178 | 32.9M | 4.49 MB | ✓ |
| googlenet | torchvision | torch | 197 | 6.6M | 4.17 MB | ✓ |
| inception_v3 | torchvision | torch | 314 | 23.8M | 5.78 MB | ✓ |
| mobilenet_v3_small | torchvision | torch | 157 | 2.5M | 3.34 MB | ✓ |
| r3d_18 (video) | torchvision | torch | 68 | 33.4M | 1.71 MB | ✓ |
| regnet_y_400mf | torchvision | torch | 270 | 4.3M | 5.23 MB | ✓ |
| resnet18 | torchvision | torch | 69 | 11.7M | 1.93 MB | ✓ |
| resnet50 | torchvision | torch | 175 | 25.6M | 4.05 MB | ✓ |
| shufflenet_v2_x0_5 | torchvision | torch | 268 | 1.4M | 3.78 MB | ✓ |
| squeezenet1_1 | torchvision | torch | 66 | 1.2M | 1.45 MB | ✓ |
| swin_t | torchvision | torch | 572 | 28.3M | 5.83 MB | ✓ |
| vit_b_16 | torchvision | torch | 142 | 86.6M | 4.74 MB | ✓ |
<!-- zoo:end -->
