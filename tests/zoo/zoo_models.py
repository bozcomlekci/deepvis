"""The model zoo: architectures deepvis must handle, all built locally (random weights or tiny configs, no downloads).

Each entry builds ``(model, inputs, native)``: the model, its positional inputs, and a function giving the model's own
output for parity checks. ``fast`` entries also run in the regular CI job; the rest run in the weekly zoo job.
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Callable

if "KERAS_BACKEND" not in os.environ:  # before Keras is imported anywhere: JAX if installed, else PyTorch
    os.environ["KERAS_BACKEND"] = "jax" if importlib.util.find_spec("jax") else "torch"


@dataclass
class Entry:
    id: str
    source: str        # where the architecture comes from: torchvision, timm, Hugging Face, JAX …
    framework: str     # what deepvis wraps: torch, flax, nnx, equinox
    build: Callable
    fast: bool = False
    needs: tuple = field(default_factory=tuple)  # importable modules it needs

    def available(self) -> bool:
        return all(importlib.util.find_spec(m) is not None for m in self.needs)


def _torch(build):
    def make():
        import torch

        torch.manual_seed(0)
        model, inputs = build(torch)
        model = model.eval()

        def native():
            with torch.no_grad():
                return model(*inputs)

        return model, inputs, native
    return make


def _tv(name, shape=(1, 3, 64, 64), **kw):
    def build(torch):
        import torchvision

        return getattr(torchvision.models, name)(weights=None, **kw), (torch.randn(*shape),)
    return _torch(build)


def _timm(name, shape=(1, 3, 64, 64), **kw):
    def build(torch):
        import timm

        return timm.create_model(name, pretrained=False, **kw), (torch.randn(*shape),)
    return _torch(build)


def _hf(config_cls, model_cls, config: dict, inputs: Callable, wrap: Callable | None = None):
    """A Hugging Face model from a tiny config; ``wrap`` adapts keyword-only signatures to positional inputs."""
    def build(torch):
        import transformers as tf

        cfg = getattr(tf, config_cls)(**config)
        m = getattr(tf, model_cls)(cfg)
        return (wrap(torch, m) if wrap else m), inputs(torch)
    return _torch(build)


def _ids(n=8, vocab=100):
    return lambda torch: (torch.randint(0, vocab, (1, n)),)


def _seq2seq(torch, m):
    class Seq2Seq(torch.nn.Module):  # model(input_ids=…, decoder_input_ids=…).logits, positionally
        def __init__(self):
            super().__init__()
            self.model = m

        def forward(self, ids, dec):
            return self.model(input_ids=ids, decoder_input_ids=dec).logits

    return Seq2Seq()


def _whisper(torch, m):
    class Whisper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = m

        def forward(self, features, dec):
            return self.model(input_features=features, decoder_input_ids=dec).logits

    return Whisper()


def _clip(torch, m):
    class Clip(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = m

        def forward(self, ids, pixels):
            return self.model(input_ids=ids, pixel_values=pixels).logits_per_image

    return Clip()


def _vae(torch):
    nn = torch.nn

    class VAE(nn.Module):  # encoder → (μ, log σ²) → z → decoder; the noise is a fixed input so outputs can be compared
        def __init__(self):
            super().__init__()
            self.enc, self.mu, self.logvar = nn.Linear(16, 8), nn.Linear(8, 2), nn.Linear(8, 2)
            self.dec = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 16), nn.Sigmoid())
            self.register_buffer("eps", torch.randn(1, 2))

        def forward(self, x):
            h = torch.relu(self.enc(x))
            mu, logvar = self.mu(h), self.logvar(h)
            z = mu + torch.exp(0.5 * logvar) * self.eps
            return self.dec(z), mu, logvar

    return VAE()


def _onnx(build, dynamo=False):
    """A PyTorch model exported to an ONNX file, by the TorchScript or the dynamo exporter; deepvis gets only the file,
    and its native output is onnxruntime's."""
    def make():
        import os
        import tempfile

        import onnxruntime as ort
        import torch

        torch.manual_seed(0)
        model, inputs = build(torch)
        path = os.path.join(tempfile.mkdtemp(prefix="deepvis-zoo-"), "model.onnx")
        torch.onnx.export(model.eval(), inputs, path, dynamo=dynamo)
        arrays = tuple(t.numpy() for t in inputs)

        def native():
            sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            return sess.run(None, {i.name: a for i, a in zip(sess.get_inputs(), arrays)})[0]

        return path, arrays, native
    return make


def _logits(torch, m, key="logits"):
    class Logits(torch.nn.Module):  # a Hugging Face model's output object → one tensor, which exporters need
        def __init__(self):
            super().__init__()
            self.m = m

        def forward(self, x):
            return getattr(self.m(x), key)

    return Logits()


def _lstm_classifier(torch):
    nn = torch.nn

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.lstm, self.head = nn.Embedding(100, 16), nn.LSTM(16, 24, batch_first=True), nn.Linear(24, 3)

        def forward(self, x):
            return self.head(self.lstm(self.emb(x))[0][:, -1])

    return Classifier(), (torch.randint(0, 100, (1, 7)),)


def _unet(torch):
    nn = torch.nn

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
            self.mid = nn.Conv2d(8, 16, 3, padding=1)
            self.up = nn.ConvTranspose2d(16, 8, 2, stride=2)
            self.out = nn.Conv2d(16, 2, 1)

        def forward(self, x):
            d = self.down(x)
            return self.out(torch.cat([d, self.up(torch.relu(self.mid(nn.functional.max_pool2d(d, 2))))], 1))

    return UNet(), (torch.randn(1, 3, 16, 16),)


def _hf_onnx(config_cls, model_cls, config, inputs, key="logits"):
    def build(torch):
        import transformers as tf

        return _logits(torch, getattr(tf, model_cls)(getattr(tf, config_cls)(**config)), key), inputs(torch)
    return build


def _keras(build):
    """A Keras 3 model (on the backend KERAS_BACKEND names) and NumPy inputs; its native output is Keras' own call."""
    def make():
        import keras

        keras.utils.set_random_seed(0)
        model, inputs = build(keras)
        return model, inputs, lambda: model(*inputs, training=False)
    return make


def _keras_cnn(keras):
    L = keras.layers
    m = keras.Sequential([keras.Input((32, 32, 3)), L.Conv2D(16, 3, activation="relu"), L.BatchNormalization(),
                          L.MaxPooling2D(), L.Conv2D(32, 3, activation="relu"), L.GlobalAveragePooling2D(),
                          L.Dropout(0.2), L.Dense(10, activation="softmax")])
    return m, (__import__("numpy").random.default_rng(0).standard_normal((1, 32, 32, 3)).astype("float32"),)


def _keras_resblock(keras):
    L = keras.layers
    inp = keras.Input((16, 16, 3), name="image")
    x = L.Conv2D(8, 3, padding="same", name="stem")(inp)
    for k in range(2):
        y = keras.Sequential([L.Conv2D(8, 3, padding="same", activation="relu"), L.BatchNormalization(),
                              L.Conv2D(8, 3, padding="same")], name=f"block{k}")(x)
        x = keras.ops.relu(L.Add()([x, y]))
    out = L.Dense(5)(L.GlobalAveragePooling2D()(x))
    return keras.Model(inp, out), (__import__("numpy").random.default_rng(0).standard_normal((1, 16, 16, 3))
                                   .astype("float32"),)


def _keras_text(keras):
    L = keras.layers
    tok = keras.Input((12,), dtype="int32")
    e = L.Embedding(100, 16)(tok)
    e = L.Bidirectional(L.LSTM(8, return_sequences=True))(e)
    a = L.MultiHeadAttention(num_heads=2, key_dim=8)(e, e)
    e = L.LayerNormalization()(L.Add()([e, a]))
    return keras.Model(tok, L.Dense(3)(L.GlobalAveragePooling1D()(e))), (
        __import__("numpy").random.default_rng(0).integers(0, 100, (1, 12)).astype("int32"),)


def _keras_app(name, shape=(64, 64, 3)):
    def build(keras):
        m = getattr(keras.applications, name)(weights=None, input_shape=shape, classes=10)
        return m, (__import__("numpy").random.default_rng(0).standard_normal((1,) + shape).astype("float32"),)
    return _keras(build)


TINY_LM = dict(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
               bos_token_id=0, eos_token_id=0, pad_token_id=0)


def _flax_cnn():
    import jax
    import jax.numpy as jnp
    import flax.linen as nn

    class CNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.relu(nn.Conv(8, (3, 3))(x))
            x = nn.avg_pool(x, (2, 2), strides=(2, 2))
            return nn.Dense(10)(x.reshape(x.shape[0], -1))

    x = jnp.ones((1, 16, 16, 3))
    m = CNN()
    v = m.init(jax.random.PRNGKey(0), x)
    return (m, v), (x,), lambda: m.apply(v, x)


def _nnx_mlp():
    import jax.numpy as jnp
    from flax import nnx

    class MLP(nnx.Module):
        def __init__(self, rngs):
            self.a, self.b = nnx.Linear(6, 16, rngs=rngs), nnx.Linear(16, 3, rngs=rngs)

        def __call__(self, x):
            return self.b(nnx.gelu(self.a(x)))

    m = MLP(nnx.Rngs(0))
    x = jnp.ones((2, 6))
    return m, (x,), lambda: m(x)


def _equinox_mlp():
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    m = eqx.nn.MLP(6, 3, width_size=16, depth=2, key=jax.random.PRNGKey(0))
    x = jnp.ones((6,))
    return m, (x,), lambda: m(x)


ZOO: list[Entry] = [
    # ---- torchvision: classification, video, segmentation
    Entry("resnet18", "torchvision", "torch", _tv("resnet18"), fast=True, needs=("torchvision",)),
    Entry("resnet50", "torchvision", "torch", _tv("resnet50"), needs=("torchvision",)),
    Entry("mobilenet_v3_small", "torchvision", "torch", _tv("mobilenet_v3_small"), fast=True, needs=("torchvision",)),
    Entry("efficientnet_b0", "torchvision", "torch", _tv("efficientnet_b0"), needs=("torchvision",)),
    Entry("convnext_tiny", "torchvision", "torch", _tv("convnext_tiny"), needs=("torchvision",)),
    Entry("regnet_y_400mf", "torchvision", "torch", _tv("regnet_y_400mf"), needs=("torchvision",)),
    Entry("densenet121", "torchvision", "torch", _tv("densenet121"), needs=("torchvision",)),
    Entry("squeezenet1_1", "torchvision", "torch", _tv("squeezenet1_1"), fast=True, needs=("torchvision",)),
    Entry("shufflenet_v2_x0_5", "torchvision", "torch", _tv("shufflenet_v2_x0_5"), needs=("torchvision",)),
    Entry("googlenet", "torchvision", "torch", _tv("googlenet", aux_logits=False, init_weights=True),
          needs=("torchvision",)),
    Entry("inception_v3", "torchvision", "torch", _tv("inception_v3", (1, 3, 75, 75), aux_logits=False,
                                                      init_weights=True), needs=("torchvision",)),
    Entry("vit_b_16", "torchvision", "torch", _tv("vit_b_16", (1, 3, 224, 224)), needs=("torchvision",)),
    Entry("swin_t", "torchvision", "torch", _tv("swin_t"), needs=("torchvision",)),
    Entry("r3d_18 (video)", "torchvision", "torch",
          _torch(lambda torch: (__import__("torchvision.models.video", fromlist=["r3d_18"]).r3d_18(weights=None),
                                (torch.randn(1, 3, 4, 32, 32),))), needs=("torchvision",)),
    Entry("fcn_resnet50 (segmentation)", "torchvision", "torch",
          _torch(lambda torch: (__import__("torchvision.models.segmentation", fromlist=["fcn_resnet50"])
                                .fcn_resnet50(weights=None, weights_backbone=None, num_classes=4),
                                (torch.randn(1, 3, 64, 64),))), needs=("torchvision",)),
    Entry("fasterrcnn_mobilenet_v3 (detection)", "torchvision", "torch",
          _torch(lambda torch: (__import__("torchvision.models.detection", fromlist=["x"])
                                .fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None, num_classes=5),
                                ([torch.rand(3, 96, 96)],))), needs=("torchvision",)),
    Entry("VAE (your own, tuple output)", "custom", "torch", _torch(lambda torch: (_vae(torch), (torch.randn(2, 16),))),
          fast=True),
    # ---- timm
    Entry("vit_tiny_patch16_224", "timm", "torch", _timm("vit_tiny_patch16_224", (1, 3, 224, 224)), needs=("timm",)),
    Entry("deit_tiny_patch16_224", "timm", "torch", _timm("deit_tiny_patch16_224", (1, 3, 224, 224)), needs=("timm",)),
    Entry("convnext_atto", "timm", "torch", _timm("convnext_atto"), needs=("timm",)),
    Entry("efficientnet_b0 (timm)", "timm", "torch", _timm("efficientnet_b0"), needs=("timm",)),
    Entry("mixer_s16_224", "timm", "torch", _timm("mixer_s16_224", (1, 3, 224, 224)), needs=("timm",)),
    Entry("resnet18 (timm)", "timm", "torch", _timm("resnet18"), fast=True, needs=("timm",)),
    # ---- Hugging Face, from tiny configs
    Entry("BERT", "Hugging Face", "torch",
          _hf("BertConfig", "BertForSequenceClassification", dict(TINY_LM, num_labels=3), _ids()), fast=True,
          needs=("transformers",)),
    Entry("DistilBERT", "Hugging Face", "torch",
          _hf("DistilBertConfig", "DistilBertModel", dict(vocab_size=100, dim=32, n_layers=2, n_heads=2, hidden_dim=64),
              _ids()), needs=("transformers",)),
    Entry("RoBERTa", "Hugging Face", "torch",
          _hf("RobertaConfig", "RobertaModel", dict(TINY_LM, pad_token_id=1), lambda t: (t.randint(2, 100, (1, 8)),)),
          needs=("transformers",)),
    Entry("GPT-2", "Hugging Face", "torch",
          _hf("GPT2Config", "GPT2LMHeadModel", dict(vocab_size=100, n_embd=32, n_layer=2, n_head=2, bos_token_id=0,
                                                    eos_token_id=0), _ids()), fast=True, needs=("transformers",)),
    Entry("GPT-NeoX", "Hugging Face", "torch",
          _hf("GPTNeoXConfig", "GPTNeoXForCausalLM", TINY_LM, _ids()), needs=("transformers",)),
    Entry("Llama", "Hugging Face", "torch",
          _hf("LlamaConfig", "LlamaForCausalLM", dict(TINY_LM, num_key_value_heads=2), _ids()), fast=True,
          needs=("transformers",)),
    Entry("Mistral", "Hugging Face", "torch",
          _hf("MistralConfig", "MistralForCausalLM", dict(TINY_LM, num_key_value_heads=2), _ids()),
          needs=("transformers",)),
    Entry("Mixtral (MoE)", "Hugging Face", "torch",
          _hf("MixtralConfig", "MixtralForCausalLM", dict(TINY_LM, num_key_value_heads=2, num_local_experts=4,
                                                          num_experts_per_tok=2), _ids()), needs=("transformers",)),
    Entry("Qwen2", "Hugging Face", "torch",
          _hf("Qwen2Config", "Qwen2ForCausalLM", dict(TINY_LM, num_key_value_heads=2), _ids()), needs=("transformers",)),
    Entry("Phi", "Hugging Face", "torch", _hf("PhiConfig", "PhiForCausalLM", TINY_LM, _ids()), needs=("transformers",)),
    Entry("Gemma", "Hugging Face", "torch",
          _hf("GemmaConfig", "GemmaForCausalLM", dict(TINY_LM, num_key_value_heads=2, head_dim=8), _ids()),
          needs=("transformers",)),
    Entry("T5 (encoder-decoder)", "Hugging Face", "torch",
          _hf("T5Config", "T5ForConditionalGeneration", dict(vocab_size=100, d_model=32, d_ff=64, d_kv=8, num_layers=2,
                                                             num_heads=4, decoder_start_token_id=0),
              lambda t: (t.randint(0, 100, (1, 8)), t.randint(0, 100, (1, 5))), _seq2seq), needs=("transformers",)),
    Entry("BART (encoder-decoder)", "Hugging Face", "torch",
          _hf("BartConfig", "BartForConditionalGeneration",
              dict(vocab_size=100, d_model=32, encoder_layers=2, decoder_layers=2, encoder_attention_heads=4,
                   decoder_attention_heads=4, encoder_ffn_dim=64, decoder_ffn_dim=64, max_position_embeddings=64),
              lambda t: (t.randint(3, 100, (1, 8)), t.randint(3, 100, (1, 5))), _seq2seq), needs=("transformers",)),
    Entry("Whisper (speech)", "Hugging Face", "torch",
          _hf("WhisperConfig", "WhisperForConditionalGeneration",
              dict(vocab_size=100, d_model=32, encoder_layers=1, decoder_layers=1, encoder_attention_heads=2,
                   decoder_attention_heads=2, encoder_ffn_dim=64, decoder_ffn_dim=64, num_mel_bins=16,
                   max_source_positions=25, max_target_positions=16, decoder_start_token_id=0, pad_token_id=0,
                   bos_token_id=0, eos_token_id=0),
              lambda t: (t.randn(1, 16, 50), t.randint(0, 100, (1, 4))), _whisper), needs=("transformers",)),
    Entry("wav2vec 2.0 (audio)", "Hugging Face", "torch",
          _hf("Wav2Vec2Config", "Wav2Vec2Model",
              dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, conv_dim=(16, 16),
                   conv_kernel=(10, 3), conv_stride=(5, 2), num_conv_pos_embeddings=16, num_conv_pos_embedding_groups=2,
                   num_feat_extract_layers=2), lambda t: (t.randn(1, 400),)), needs=("transformers",)),
    Entry("ViT", "Hugging Face", "torch",
          _hf("ViTConfig", "ViTForImageClassification",
              dict(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                   intermediate_size=64, num_labels=5), lambda t: (t.randn(1, 3, 32, 32),)), needs=("transformers",)),
    Entry("Swin", "Hugging Face", "torch",
          _hf("SwinConfig", "SwinModel", dict(image_size=32, patch_size=4, embed_dim=16, depths=[1, 1], num_heads=[2, 2],
                                              window_size=4), lambda t: (t.randn(1, 3, 32, 32),)),
          needs=("transformers",)),
    Entry("CLIP (image + text)", "Hugging Face", "torch",
          _hf("CLIPConfig", "CLIPModel",
              dict(text_config=dict(vocab_size=100, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                    num_attention_heads=2, max_position_embeddings=16),
                   vision_config=dict(image_size=32, patch_size=8, hidden_size=32, intermediate_size=64,
                                      num_hidden_layers=1, num_attention_heads=2), projection_dim=16),
              lambda t: (t.randint(0, 100, (2, 6)), t.randn(1, 3, 32, 32)), _clip), needs=("transformers",)),
    Entry("Mamba (state space)", "Hugging Face", "torch",
          _hf("MambaConfig", "MambaForCausalLM", dict(vocab_size=100, hidden_size=32, num_hidden_layers=2, state_size=4,
                                                      expand=2, conv_kernel=4), _ids()), needs=("transformers",)),
    # ---- ONNX files (exported here from PyTorch, then given to deepvis as files)
    Entry("ResNet-18 · TorchScript exporter", "ONNX", "onnx",
          _onnx(lambda torch: (__import__("torchvision").models.resnet18(weights=None), (torch.randn(1, 3, 64, 64),))),
          fast=True, needs=("onnx", "onnxruntime", "torchvision")),
    Entry("ResNet-18 · dynamo exporter", "ONNX", "onnx",
          _onnx(lambda torch: (__import__("torchvision").models.resnet18(weights=None), (torch.randn(1, 3, 64, 64),)),
                dynamo=True), needs=("onnx", "onnxruntime", "onnxscript", "torchvision")),
    Entry("MobileNetV3 · TorchScript exporter", "ONNX", "onnx",
          _onnx(lambda torch: (__import__("torchvision").models.mobilenet_v3_small(weights=None),
                               (torch.randn(1, 3, 64, 64),))), needs=("onnx", "onnxruntime", "torchvision")),
    Entry("U-Net style (transposed conv, skip) · dynamo exporter", "ONNX", "onnx", _onnx(_unet, dynamo=True),
          needs=("onnx", "onnxruntime", "onnxscript")),
    Entry("LSTM classifier · TorchScript exporter", "ONNX", "onnx", _onnx(_lstm_classifier), fast=True,
          needs=("onnx", "onnxruntime")),
    Entry("Transformer encoder · dynamo exporter", "ONNX", "onnx",
          _onnx(lambda torch: (torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(32, 4, 64, batch_first=True,
                                                                                            dropout=0.0), 2,
                                                           enable_nested_tensor=False), (torch.randn(1, 10, 32),)),
                dynamo=True), needs=("onnx", "onnxruntime", "onnxscript")),
    Entry("GPT-2 · dynamo exporter", "ONNX", "onnx",
          _onnx(_hf_onnx("GPT2Config", "GPT2LMHeadModel", dict(vocab_size=100, n_embd=32, n_layer=2, n_head=2,
                                                               bos_token_id=0, eos_token_id=0), _ids()), dynamo=True),
          needs=("onnx", "onnxruntime", "onnxscript", "transformers")),
    Entry("BERT · dynamo exporter", "ONNX", "onnx",
          _onnx(_hf_onnx("BertConfig", "BertModel", TINY_LM, _ids(9), "last_hidden_state"), dynamo=True),
          needs=("onnx", "onnxruntime", "onnxscript", "transformers")),
    Entry("ViT · TorchScript exporter", "ONNX", "onnx",
          _onnx(_hf_onnx("ViTConfig", "ViTForImageClassification",
                         dict(image_size=32, patch_size=8, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                              intermediate_size=64, num_labels=5), lambda t: (t.randn(1, 3, 32, 32),))),
          needs=("onnx", "onnxruntime", "transformers")),
    # ---- Keras 3 (JAX or PyTorch backend: KERAS_BACKEND)
    Entry("Keras Sequential CNN", "Keras", "keras", _keras(_keras_cnn), fast=True, needs=("keras",)),
    Entry("Keras functional (residual blocks)", "Keras", "keras", _keras(_keras_resblock), fast=True,
          needs=("keras",)),
    Entry("Keras text (Bidirectional LSTM, attention)", "Keras", "keras", _keras(_keras_text), fast=True,
          needs=("keras",)),
    Entry("Keras MobileNetV3Small", "Keras", "keras", _keras_app("MobileNetV3Small"), needs=("keras",)),
    Entry("Keras ResNet50", "Keras", "keras", _keras_app("ResNet50"), needs=("keras",)),
    Entry("Keras EfficientNetB0", "Keras", "keras", _keras_app("EfficientNetB0"), needs=("keras",)),
    Entry("Keras ConvNeXtTiny", "Keras", "keras", _keras_app("ConvNeXtTiny"), needs=("keras",)),
    # ---- JAX
    Entry("Flax linen CNN", "JAX", "flax", _flax_cnn, fast=True, needs=("flax",)),
    Entry("Flax nnx MLP", "JAX", "nnx", _nnx_mlp, fast=True, needs=("flax",)),
    Entry("Equinox MLP", "JAX", "equinox", _equinox_mlp, fast=True, needs=("equinox",)),
]
