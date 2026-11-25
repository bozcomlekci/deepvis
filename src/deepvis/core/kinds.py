"""Canonical layer kinds shared by every backend.

Backends report framework-specific class names (``Dense``, ``nnx.Linear``,
``eqx.nn.Linear`` ...). Everything downstream (colors, equations, neuron views,
neuron-level edits) works on the canonical kinds defined here.
"""

from __future__ import annotations

from dataclasses import dataclass

_ALIASES = {
    "Dense": "Linear",
    "DenseGeneral": "Linear",
    "LinearGeneral": "Linear",
    "NonDynamicallyQuantizableLinear": "Linear",
    "Embed": "Embedding",
    "MultiHeadDotProductAttention": "MultiheadAttention",
    "MultiHeadAttention": "MultiheadAttention",
    "SelfAttention": "MultiheadAttention",
    "RMSNorm": "RMSNorm",
    "LayerNorm": "LayerNorm",
    "GroupNorm": "GroupNorm",
    "BatchNorm": "BatchNorm",
    "BatchNorm1d": "BatchNorm",
    "BatchNorm2d": "BatchNorm",
    "BatchNorm3d": "BatchNorm",
    "Dropout": "Dropout",
    "Dropout1d": "Dropout",
    "Dropout2d": "Dropout",
    "AlphaDropout": "Dropout",
    "Identity": "Identity",
    # Keras 3 layer classes
    "Conv1D": "Conv1d", "Conv2D": "Conv2d", "Conv3D": "Conv3d", "DepthwiseConv1D": "Conv1d",
    "DepthwiseConv2D": "Conv2d", "SeparableConv1D": "Conv1d", "SeparableConv2D": "Conv2d",
    "Conv1DTranspose": "ConvTranspose1d", "Conv2DTranspose": "ConvTranspose2d", "Conv3DTranspose": "ConvTranspose3d",
    "BatchNormalization": "BatchNorm", "LayerNormalization": "LayerNorm", "GroupNormalization": "GroupNorm",
    "UnitNormalization": "Normalize", "MaxPooling1D": "MaxPool1d", "MaxPooling2D": "MaxPool2d",
    "MaxPooling3D": "MaxPool3d", "AveragePooling1D": "AvgPool1d", "AveragePooling2D": "AvgPool2d",
    "AveragePooling3D": "AvgPool3d", "GlobalAveragePooling1D": "AdaptiveAvgPool1d",
    "GlobalAveragePooling2D": "AdaptiveAvgPool2d", "GlobalAveragePooling3D": "AdaptiveAvgPool3d",
    "GlobalMaxPooling1D": "AdaptiveMaxPool1d", "GlobalMaxPooling2D": "AdaptiveMaxPool2d",
    "GlobalMaxPooling3D": "AdaptiveMaxPool3d", "SimpleRNN": "RNN", "SpatialDropout1D": "Dropout",
    "SpatialDropout2D": "Dropout", "SpatialDropout3D": "Dropout", "GaussianDropout": "Dropout",
    "GaussianNoise": "Identity", "ActivityRegularization": "Identity", "Add": "add", "Subtract": "sub",
    "Multiply": "mul", "Average": "mean", "Maximum": "max", "Minimum": "min", "Concatenate": "Concat",
    "Dot": "matmul", "UpSampling1D": "Upsample", "UpSampling2D": "Upsample", "UpSampling3D": "Upsample",
    "Permute": "Transpose", "EinsumDense": "Linear", "ZeroPadding1D": "pad", "ZeroPadding2D": "pad",
    "ZeroPadding3D": "pad", "Cropping1D": "getitem", "Cropping2D": "getitem", "Cropping3D": "getitem",
    # torch builtins whose names contain a pattern below but mean something else
    "LogSigmoid": "LogSigmoid",
    "Tanhshrink": "Tanhshrink",
}

# function / primitive names -> canonical kinds
FUNCTION_KINDS = {
    "relu": "ReLU", "relu6": "ReLU", "leaky_relu": "LeakyReLU", "gelu": "GELU", "silu": "SiLU",
    "swish": "SiLU", "sigmoid": "Sigmoid", "logistic": "Sigmoid", "tanh": "Tanh", "softmax": "Softmax",
    "log_softmax": "LogSoftmax", "elu": "ELU", "softplus": "Softplus", "mish": "Mish",
    "hardtanh": "Hardtanh", "exp": "exp", "log": "log", "abs": "abs", "square": "square",
    "add": "add", "iadd": "add", "__add__": "add", "__radd__": "add", "sub": "sub", "__sub__": "sub",
    "rsub": "sub", "mul": "mul", "__mul__": "mul", "imul": "mul", "__rmul__": "mul", "div": "div",
    "truediv": "div", "__truediv__": "div", "neg": "neg", "matmul": "matmul", "bmm": "matmul",
    "mm": "matmul", "einsum": "matmul", "dot_general": "matmul", "linear": "Linear",
    "flatten": "Flatten", "view": "Reshape", "reshape": "Reshape", "permute": "Transpose",
    "transpose": "Transpose", "cat": "Concat", "concat": "Concat", "concatenate": "Concat",
    "stack": "Concat", "mean": "mean", "sum": "sum", "amax": "max", "max": "max",
    "dropout": "Dropout", "layer_norm": "LayerNorm", "conv2d": "Conv2d", "conv1d": "Conv1d",
    "max_pool2d": "MaxPool2d", "avg_pool2d": "AvgPool2d", "adaptive_avg_pool2d": "AdaptiveAvgPool2d",
    "getitem": "getitem", "sqrt": "sqrt", "rsqrt": "rsqrt", "pow": "pow", "integer_pow": "pow",
    "erf": "erf", "reduce_sum": "sum", "reduce_max": "max",
    "scaled_dot_product_attention": "Attention", "dot_product_attention": "Attention",
    "T": "Transpose", "mT": "Transpose", "t": "Transpose", "hardswish": "Hardswish", "hardsigmoid": "Hardsigmoid",
    "hard_swish": "Hardswish", "hard_sigmoid": "Hardsigmoid", "selu": "SELU", "celu": "CELU", "softsign": "Softsign",
    "prelu": "PReLU", "glu": "GLU", "normalize": "Normalize", "interpolate": "Upsample", "upsample": "Upsample",
    "pixel_shuffle": "PixelShuffle",
    "batch_norm": "BatchNorm", "group_norm": "GroupNorm", "rms_norm": "RMSNorm", "instance_norm": "InstanceNorm2d",
    "chunk": "split", "unbind": "unbind", "baddbmm": "matmul",
}

ELEMENTWISE_KINDS = {
    "ReLU", "LeakyReLU", "GELU", "SiLU", "Sigmoid", "Tanh", "ELU", "Softplus", "Mish", "Hardtanh", "Hardswish",
    "Hardsigmoid", "ReLU6", "PReLU", "SELU", "CELU", "Softsign",
    "exp", "log", "abs", "square", "neg", "sqrt", "rsqrt", "pow", "erf", "Dropout", "Identity",
}
ACTIVATION_KINDS = ELEMENTWISE_KINDS - {"Dropout", "Identity"} | {"Softmax", "LogSoftmax"}
# kinds whose weight matrix maps "in" neurons to "out" neurons
DENSE_KINDS = {"Linear"}
CONV_KINDS = {"Conv1d", "Conv2d", "Conv3d"}
TRANSPOSED_CONV_KINDS = {"ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d"}
NORM_KINDS = {"LayerNorm", "RMSNorm", "BatchNorm", "GroupNorm", "InstanceNorm2d"}


def canonical_kind(type_name: str, attrs: dict | None = None) -> str:
    attrs = attrs or {}
    if type_name in ("Conv", "ConvLocal"):
        ks = attrs.get("kernel_size")
        n = len(ks) if isinstance(ks, (tuple, list)) else 2
        return f"Conv{n}d"
    if type_name == "Lambda" and attrs.get("fn_name"):
        return function_kind(attrs["fn_name"])
    if type_name == "Activation" and isinstance(attrs.get("activation"), str):  # Keras: Activation("relu")
        return function_kind(attrs["activation"])
    if type_name in _ALIASES:
        return _ALIASES[type_name]
    return _by_name(type_name) or type_name


# third-party layer classes (HuggingFace, timm, ...) recognised by name: first match wins
_NAME_PATTERNS = [
    ("RMSNorm", "RMSNorm"), ("T5LayerNorm", "RMSNorm"), ("LayerNorm", "LayerNorm"), ("GroupNorm", "GroupNorm"),
    ("BatchNorm", "BatchNorm"), ("RotaryEmbedding", "RotaryEmbedding"), ("GELU", "GELU"), ("Gelu", "GELU"),
    ("SiLU", "SiLU"), ("Swish", "SiLU"), ("Mish", "Mish"), ("Sigmoid", "Sigmoid"), ("Tanh", "Tanh"),
]
_ACT_SUFFIXES = ("Activation", "Act", "Layer", "Module")


def _by_name(type_name: str) -> str | None:
    if type_name in FUNCTION_KINDS.values():
        return None
    for pat, kind in _NAME_PATTERNS:
        if pat in type_name:
            return kind
    for suf in _ACT_SUFFIXES:  # e.g. "ReLUActivation" -> "ReLU"
        if type_name.endswith(suf) and type_name[: -len(suf)] in ELEMENTWISE_KINDS:
            return type_name[: -len(suf)]
    return None


def function_kind(name: str) -> str:
    name = str(name).strip("_") if str(name) not in FUNCTION_KINDS else str(name)
    return FUNCTION_KINDS.get(name, FUNCTION_KINDS.get(name.lower(), name))


@dataclass
class Role:
    """Where a semantically meaningful parameter lives and how its axes map to neurons."""

    name: str  # local parameter name, e.g. "weight" / "kernel"
    out_axis: int | None = None
    in_axis: int | None = None


def infer_roles(kind: str, param_shapes: dict[str, tuple], framework: str) -> dict[str, Role]:
    """Map canonical roles (weight, bias, scale, ...) to local param names and neuron axes.

    torch & equinox store Linear weights as (out, in); flax/nnx store kernels as (in, out).
    Conv: torch/equinox (out, in, *k); flax/nnx (*k, in, out).
    """
    names = set(param_shapes)
    roles: dict[str, Role] = {}
    channels_last = framework in ("flax", "nnx", "keras")
    wname = next((n for n in ("weight", "kernel", "W", "w") if n in names), None)
    bname = next((n for n in ("bias", "b", "beta") if n in names), None)
    if kind in TRANSPOSED_CONV_KINDS and wname is not None and not channels_last:
        roles["weight"] = Role(wname, out_axis=1, in_axis=0)
        if bname is not None:
            roles["bias"] = Role(bname, out_axis=0)
        return roles
    if kind in DENSE_KINDS or kind in CONV_KINDS or kind in TRANSPOSED_CONV_KINDS:
        if wname is not None:
            nd = len(param_shapes[wname])
            if channels_last:
                roles["weight"] = Role(wname, out_axis=nd - 1, in_axis=nd - 2)
            else:
                roles["weight"] = Role(wname, out_axis=0, in_axis=1 if nd > 1 else None)
        if bname is not None:
            roles["bias"] = Role(bname, out_axis=0)
    elif kind in NORM_KINDS:
        sname = next((n for n in ("weight", "scale", "gamma") if n in names), None)
        if sname:
            roles["scale"] = Role(sname, out_axis=0)
        if bname:
            roles["bias"] = Role(bname, out_axis=0)
        for stat in ("running_mean", "running_var", "mean", "var", "moving_mean", "moving_variance"):
            if stat in names:
                roles[stat] = Role(stat, out_axis=0)
    elif kind == "Embedding":
        ename = next((n for n in ("weight", "embedding", "embeddings") if n in names), None)
        if ename:
            roles["weight"] = Role(ename, out_axis=len(param_shapes[ename]) - 1, in_axis=0)
    else:
        if wname:
            roles["weight"] = Role(wname)
        if bname:
            roles["bias"] = Role(bname)
    return roles
