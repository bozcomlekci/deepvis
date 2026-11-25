"""A transformer: attention maps, the residual stream, and activation-level interventions."""

import os

import torch
import torch.nn as nn

import deepvis as dv

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)


class TinyTransformer(nn.Module):
    def __init__(self, vocab=64, d=32, heads=4, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.randn(32, d) * 0.02)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 64, dropout=0.0, batch_first=True),
                                             layers)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        return self.head(self.encoder(self.embed(ids) + self.pos[: ids.shape[1]]))


model = TinyTransformer().eval()
ids = torch.randint(0, 64, (2, 12))
net = dv.wrap(model, sample_input=ids)
print(net.equation())                                          # residual stream made explicit
print(net.equation(depth=2))                                   # layers collapsed into blocks

tr = net.trace(ids)
for name, att in tr.attention.items():
    print(name, "attention", att.shape)                      # (batch, heads, query, key)

residual = [n.id for n in net.graph.compute_nodes() if n.kind == "add"]
print("residual adds:", residual)
base = net(ids).detach()
with net.intervene("encoder.layers.1.self_attn", dv.ablate()):  # knock out a whole attention layer
    print("Δlogits without layer-1 attention:", (net(ids) - base).norm().item())
patched = dv.patch(net, ids, torch.randint(0, 64, (2, 12)), at="encoder.layers.0.norm2")
print("activation patching recovery:", round(patched.recovery(), 3))
print("saved", net.save_html(os.path.join(OUT, "03_transformer.html"), ids))
