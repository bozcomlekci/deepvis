"""The logit lens: what a language model would predict if it stopped after each block.

Every block's output (the residual stream) is sent through the model's own final norm and unembedding —
the modules deepvis finds in the graph — so the last row reproduces the real prediction exactly.

>>> lens = net.logit_lens(ids, tokenizer=tok)
>>> print(lens.table())           # layers × positions, top token per cell
>>> lens.probs(" Paris")          # how that token's probability builds up, layer by layer (last position)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.graph import ModelGraph
from ..core.kinds import NORM_KINDS

if TYPE_CHECKING:  # pragma: no cover
    from ..net import Net

_PASS = {"Dropout", "Identity", "Reshape", "getitem", "Flatten", "Transpose"}


@dataclass
class LogitLens:
    layers: list[str]                 # "embeddings", then one entry per block
    positions: list[int]              # which input positions are shown
    top_ids: np.ndarray               # (layers, positions, k)
    top_p: np.ndarray                 # (layers, positions, k)
    probs_all: np.ndarray | None = None   # (layers, positions, vocab) when kept
    tokens: list[str] | None = None   # input token strings (all positions)
    names: Any = None                 # tokenizer, for labels
    meta: dict = field(default_factory=dict)

    def label(self, i: int) -> str:
        if self.names is None:
            return f"#{int(i)}"
        from ..viz.html.payload import token_strings

        return token_strings(self.names, [int(i)])[0]

    def table(self, width: int = 9) -> str:
        head = "layer".ljust(14) + "".join((self.tokens[p] if self.tokens else str(p))[:width].ljust(width + 1)
                                           for p in self.positions)
        rows = [head]
        for li, name in enumerate(self.layers):
            cells = "".join(self.label(self.top_ids[li, j, 0])[:width].ljust(width + 1)
                            for j in range(len(self.positions)))
            rows.append(name[:13].ljust(14) + cells)
        return "\n".join(rows)

    def probs(self, token, position: int = -1) -> np.ndarray:
        """Probability of ``token`` (id, or a string the tokenizer knows) after each layer at ``position``."""
        if self.probs_all is None:
            raise ValueError("call logit_lens(..., keep_probs=True) to query arbitrary tokens")
        if isinstance(token, str):
            if self.names is None or not hasattr(self.names, "encode"):
                raise ValueError("a string token needs a tokenizer")
            token = self.names.encode(token, add_special_tokens=False)[0]
        j = self.positions.index(self.positions[position]) if position < 0 else self.positions.index(position)
        return self.probs_all[:, j, int(token)]

    def to_dict(self) -> dict:
        return {"layers": self.layers, "positions": self.positions, "tokens": self.tokens,
                "nodes": self.meta.get("blocks", []), "norm": self.meta.get("norm"), "head": self.meta.get("head"),
                "top": [[[{"id": int(i), "p": float(p), "label": self.label(i) if self.names is not None else None}
                          for i, p in zip(self.top_ids[li, j], self.top_p[li, j])]
                         for j in range(len(self.positions))] for li in range(len(self.layers))]}

    def __repr__(self) -> str:
        return f"LogitLens({len(self.layers)} layers × {len(self.positions)} positions)\n{self.table()}"


# ------------------------------------------------------------------------------------------ discovery
def _back(g: ModelGraph, nid: str) -> str | None:
    p = g.pred(nid)
    return p[0] if len(p) == 1 else None


def find_head(g: ModelGraph) -> tuple[str | None, str | None]:
    """(final norm, unembedding) feeding the output: output ← [reshape/getitem] ← Linear ← [...] ← Norm."""
    outs = [e.src for e in g.edges if g.nodes[e.dst].op == "output"]
    cur = outs[0] if outs else None
    while cur is not None and g.nodes[cur].kind in _PASS:
        cur = _back(g, cur)
    if cur is None or g.nodes[cur].kind != "Linear" or "weight" not in g.nodes[cur].roles:
        return None, None
    head = cur
    cur = _back(g, head)
    while cur is not None and g.nodes[cur].kind in _PASS:
        cur = _back(g, cur)
    norm = cur if cur is not None and g.nodes[cur].kind in NORM_KINDS else None
    return norm, head


def find_blocks(g: ModelGraph) -> list[str]:
    """The repeated blocks: the container with the most (≥ 2) same-type container children."""
    best: list[str] = []
    for c in g.nodes.values():
        if not c.is_container:
            continue
        kids = [g.nodes[k] for k in c.children if k in g.nodes and g.nodes[k].is_container]
        by_type: dict[str, list[str]] = {}
        for k in kids:
            by_type.setdefault(k.type_name, []).append(k.id)
        for ids in by_type.values():
            if len(ids) >= 2 and len(ids) > len(best):
                best = ids
    order = {nid: i for i, nid in enumerate(g.topo_order())}
    return sorted(best, key=lambda b: min((order.get(d, 1 << 30) for d in g.descendants(b)), default=0))


def _block_io(g: ModelGraph, block: str) -> tuple[str | None, str | None]:
    """(node feeding the block from outside, the block's last node with a consumer outside it)."""
    inside = {d for d in g.descendants(block) if g.nodes[d].is_compute}
    order = [n for n in g.topo_order() if n in inside]
    feeds = [p for n in order for p in g.pred(n) if p not in inside and g.nodes[p].op != "param"]
    outs = [n for n in order if any(s not in inside for s in g.succ(n))]
    return (feeds[0] if feeds else None), (outs[-1] if outs else None)


# ------------------------------------------------------------------------------------------ the lens
def top_k(p: np.ndarray, k: int) -> np.ndarray:
    """Indices of each row's ``k`` largest entries, largest first — without sorting a whole 50k-token vocabulary."""
    if k >= p.shape[-1]:
        return np.argsort(-p, axis=-1)[..., :k]
    part = np.argpartition(-p, k - 1, axis=-1)[..., :k]
    order = np.argsort(-np.take_along_axis(p, part, -1), axis=-1, kind="stable")
    return np.take_along_axis(part, order, -1)


def logit_lens(net: "Net", *inputs, sample: int = 0, k: int = 5, positions=None, max_positions: int = 32,
               tokenizer=None, keep_probs: bool | None = None, trace=None, lens_fn=None) -> LogitLens:
    """Top-``k`` predictions after the embeddings and after every block (torch models with a norm + unembedding
    head, e.g. GPT-2, Llama, Mistral, …). ``positions`` defaults to the last ``max_positions``; the full
    probabilities are kept (for :meth:`LogitLens.probs`) when they fit in ~80 MB unless ``keep_probs`` says.

    ``lens_fn(h) -> logits`` maps a block's hidden states (positions × width) to vocabulary logits — needed for JAX,
    e.g. ``lambda h: model.apply(variables, h, method=Model.unembed)``; torch models find their final norm + head."""
    from ..errors import ModelError

    if net.framework != "torch" and lens_fn is None:
        raise ModelError("for a JAX model the logit lens needs to know how to read hidden states out as logits.",
                         "pass lens_fn=lambda h: model.apply(variables, h, method=Model.unembed) — the final norm and "
                         "the unembedding applied to (positions, width) hidden states.")
    g = net.graph
    norm, head = find_head(g) if lens_fn is None else (None, "lens_fn")
    if head is None:
        raise ModelError("no language-model head found (the output should come from a Linear layer over the vocabulary).",
                         "use the model with its head, e.g. transformers.AutoModelForCausalLM rather than AutoModel.")
    blocks = find_blocks(g)
    if not blocks:
        raise ModelError("no repeated blocks found (the lens reads the output of each block of a stack).",
                         "the logit lens is for transformer-style models: a stack of ≥ 2 blocks of the same class.")
    stages = []
    first_in, _ = _block_io(g, blocks[0])
    if first_in is not None:
        stages.append(("embeddings", first_in))
    for b in blocks:
        _, out = _block_io(g, b)
        if out is not None:
            stages.append((g.nodes[b].label or b, out))
    if trace is not None and all(nid in trace.acts for _, nid in stages):
        tr = trace  # reuse a trace that already recorded every block output
    else:
        tr = net.trace(*inputs, record=[nid for _, nid in stages], capture_attention=False)
    if lens_fn is None:
        import torch

        model = net.backend.model
        head_mod = model.get_submodule(g.nodes[head].module or head)
        norm_mod = model.get_submodule(g.nodes[norm].module or norm) if norm else None
        p_ = next(iter(head_mod.parameters()))

        def read_out(x):
            with torch.no_grad():
                t = torch.as_tensor(x, dtype=p_.dtype, device=p_.device)
                return head_mod(norm_mod(t) if norm_mod is not None else t).float().cpu().numpy()
    else:
        def read_out(x):
            return np.asarray(lens_fn(x), dtype=np.float32)
    tok = tokenizer if tokenizer is not None else getattr(net, "tokenizer", None)
    tops_i, tops_p, keep, names = [], [], [], []
    pos_idx = None
    for name, nid in stages:
        a = tr.acts.get(nid)
        if a is None:
            continue
        h = np.asarray(a)
        h = h[min(sample, h.shape[0] - 1)] if h.ndim == 3 else h.reshape(-1, h.shape[-1])
        if pos_idx is None:
            T = h.shape[0]
            pos_idx = list(positions) if positions is not None else list(range(max(0, T - max_positions), T))
        logits = read_out(h[pos_idx]).astype(np.float64)
        probs = np.exp(logits - logits.max(-1, keepdims=True))
        probs /= probs.sum(-1, keepdims=True)
        top = top_k(probs, k)
        tops_i.append(top)
        tops_p.append(np.take_along_axis(probs, top, -1))
        if keep_probs is None:
            keep_probs = len(stages) * probs.size * 4 <= 80 << 20
        if keep_probs:
            keep.append(probs)
        names.append(name)
    tokens = None
    if tok is not None:
        from ..viz.html.payload import token_strings

        ins = [a for a in tr.args if hasattr(a, "shape")]
        ids = np.asarray(ins[0].detach().cpu() if hasattr(ins[0], "detach") else ins[0])  # torch or jax
        ids = ids[min(sample, ids.shape[0] - 1)] if ids.ndim >= 2 else ids
        try:
            tokens = token_strings(tok, ids)
        except Exception:  # noqa: BLE001
            tokens = None
    return LogitLens(names, pos_idx or [], np.stack(tops_i), np.stack(tops_p),
                     np.stack(keep) if keep_probs else None, tokens, tok,
                     {"norm": norm, "head": head, "stages": [nid for _, nid in stages],
                      "blocks": ([first_in] if first_in is not None else []) + list(blocks)})
