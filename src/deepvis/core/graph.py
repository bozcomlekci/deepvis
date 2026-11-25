"""Framework-agnostic intermediate representation of a network.

A :class:`ModelGraph` holds two structures over the same nodes:

* a **module tree** (``parent`` / ``children``) — containers such as ``Sequential``
  or transformer blocks group their sub-modules;
* a **dataflow DAG** (``edges``) between *compute* nodes — leaf modules, functional
  ops (residual ``+``, ``softmax`` ...), inputs, outputs and free parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from .kinds import Role

COMPUTE_OPS = ("input", "module", "function", "param", "output")


@dataclass
class ParamInfo:
    path: str  # global path, e.g. "blocks.0.fc.weight"
    name: str  # local name, e.g. "weight"
    module: str  # owning module path
    shape: tuple
    dtype: str
    trainable: bool = True

    @property
    def numel(self) -> int:
        n = 1
        for s in self.shape:
            n *= int(s)
        return n


@dataclass
class Node:
    id: str
    kind: str
    type_name: str
    op: str  # input | module | function | param | output | container
    parent: str | None = None
    children: list[str] = field(default_factory=list)
    params: dict[str, ParamInfo] = field(default_factory=dict)
    roles: dict[str, Role] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)
    in_shapes: list = field(default_factory=list)
    out_shapes: list = field(default_factory=list)
    module: str | None = None  # module whose weights this call uses (weight sharing)
    label: str | None = None

    @property
    def is_container(self) -> bool:
        return self.op == "container"

    @property
    def is_compute(self) -> bool:
        return self.op in COMPUTE_OPS

    @property
    def n_params(self) -> int:
        """Trainable parameters (buffers such as BatchNorm running statistics are not counted)."""
        return sum(p.numel for p in self.params.values() if p.trainable)

    @property
    def display(self) -> str:
        return self.label or self.id or "<root>"

    def role_param(self, role: str) -> ParamInfo | None:
        r = self.roles.get(role)
        return self.params.get(r.name) if r else None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "type": self.type_name,
            "op": self.op,
            "parent": self.parent,
            "children": list(self.children),
            "label": self.display,
            "module": self.module,
            "params": [
                {"name": p.name, "path": p.path, "shape": list(p.shape), "dtype": p.dtype,
                 "numel": p.numel, "trainable": p.trainable}
                for p in self.params.values()
            ],
            "roles": {k: {"name": r.name, "out_axis": r.out_axis, "in_axis": r.in_axis}
                      for k, r in self.roles.items()},
            "attrs": _jsonable(self.attrs),
            "in_shapes": [list(s) if s is not None else None for s in self.in_shapes],
            "out_shapes": [list(s) if s is not None else None for s in self.out_shapes],
        }


@dataclass
class Edge:
    src: str
    dst: str
    shape: tuple | None = None
    implicit: bool = False  # inferred from execution order rather than traced dataflow


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
    except ImportError:  # pragma: no cover
        pass
    return str(obj)


class ModelGraph:
    def __init__(self, name: str = "model", framework: str = ""):
        self.name = name
        self.framework = framework
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self._edge_keys: set[tuple[str, str]] = set()
        self._cache: dict = {}  # topological order, adjacency — rebuilt when nodes or edges change

    # ----------------------------------------------------------------- caching
    def _cached(self, name: str, build):
        """``build()`` once per state of the graph: big graphs ask for their order and neighbours thousands of times
        (one equation per layer), and recomputing them each time is quadratic."""
        key = (id(self.nodes), len(self.nodes), id(self.edges), len(self.edges))
        c = self.__dict__.get("_cache")
        if c is None or c.get("_key") != key:
            c = self._cache = {"_key": key}
        if name not in c:
            c[name] = build()
        return c[name]

    def changed(self) -> None:
        """Forget the cached order and neighbours (after changing a node's ``op`` in place)."""
        self._cache = {}

    def _adjacency(self) -> tuple[dict, dict]:
        def build():
            succ: dict[str, list[str]] = {}
            pred: dict[str, list[str]] = {}
            for e in self.edges:
                succ.setdefault(e.src, []).append(e.dst)
                pred.setdefault(e.dst, []).append(e.src)
            return succ, pred
        return self._cached("adjacency", build)

    # ------------------------------------------------------------------ build
    def add_node(self, node: Node) -> Node:
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        if node.parent is not None and node.parent in self.nodes:
            par = self.nodes[node.parent]
            if node.id not in par.children:
                par.children.append(node.id)
        return node

    def add_edge(self, src: str, dst: str, shape=None, implicit: bool = False) -> None:
        if src == dst or (src, dst) in self._edge_keys:
            return
        self._edge_keys.add((src, dst))
        self.edges.append(Edge(src, dst, tuple(shape) if shape is not None else None, implicit))

    def link_children(self) -> None:
        """(Re)build ``children`` lists from ``parent`` pointers, preserving insertion order."""
        for n in self.nodes.values():
            n.children = []
        for n in self.nodes.values():
            if n.parent is not None and n.parent in self.nodes and n.parent != n.id:
                self.nodes[n.parent].children.append(n.id)

    # ----------------------------------------------------------------- queries
    def __getitem__(self, key: str) -> Node:
        return self.nodes[key]

    def __contains__(self, key: str) -> bool:
        return key in self.nodes

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def root(self) -> Node | None:
        return self.nodes.get("")

    def compute_nodes(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.is_compute]

    def containers(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.is_container]

    def inputs(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.op == "input"]

    def outputs(self) -> list[Node]:
        return [n for n in self.nodes.values() if n.op == "output"]

    def succ(self, nid: str) -> list[str]:
        return list(self._adjacency()[0].get(nid, ()))

    def pred(self, nid: str) -> list[str]:
        return list(self._adjacency()[1].get(nid, ()))

    def ancestors(self, nid: str) -> list[str]:
        out, cur = [], self.nodes[nid].parent
        while cur is not None and cur in self.nodes:
            out.append(cur)
            nxt = self.nodes[cur].parent
            if nxt == cur:
                break
            cur = nxt
        return out

    def depth(self, nid: str) -> int:
        return len([a for a in self.ancestors(nid) if a != ""])

    def descendants(self, nid: str) -> list[str]:
        out, stack = [], list(self.nodes[nid].children)
        while stack:
            c = stack.pop(0)
            out.append(c)
            stack.extend(self.nodes[c].children)
        return out

    def in_subtree(self, nid: str, root: str) -> bool:
        return nid == root or root == "" or root in self.ancestors(nid)

    def topo_order(self) -> list[str]:
        """Topological order of compute nodes (insertion order breaks ties)."""
        return list(self._cached("topo", self._topo_order))

    def _topo_order(self) -> list[str]:
        import heapq

        ids = [n.id for n in self.compute_nodes()]
        index = {k: i for i, k in enumerate(ids)}
        indeg = {k: 0 for k in ids}
        adj: dict[str, list[str]] = {k: [] for k in ids}
        for e in self.edges:
            if e.src in indeg and e.dst in indeg:
                adj[e.src].append(e.dst)
                indeg[e.dst] += 1
        ready = [(index[k], k) for k in ids if indeg[k] == 0]  # a heap on insertion order: ties stay stable
        heapq.heapify(ready)
        order = []
        while ready:
            _, k = heapq.heappop(ready)
            order.append(k)
            for d in adj[k]:
                indeg[d] -= 1
                if indeg[d] == 0:
                    heapq.heappush(ready, (index[d], d))
        if len(order) < len(ids):  # cycle: fall back to insertion order for the rest
            done = set(order)
            order += [k for k in ids if k not in done]
        return order

    def param_nodes(self) -> list[Node]:
        """Compute nodes that own parameters, in topological order."""
        return [self.nodes[k] for k in self.topo_order() if self.nodes[k].params]

    def max_depth(self) -> int:
        return max((self.depth(n.id) for n in self.nodes.values() if n.is_compute), default=0)

    # --------------------------------------------------------- transformations
    def collapse(self, depth: int) -> "ModelGraph":
        """Return a graph where containers at ``depth`` become opaque compute nodes.

        depth=0 keeps only the top-level children of the root; every descendant of a
        collapsed container is folded into it and edges are re-routed.
        """
        rep: dict[str, str] = {}
        for n in self.nodes.values():
            if not n.is_compute:
                continue
            chain = [a for a in reversed(self.ancestors(n.id)) if a != ""]  # outermost first
            target = n.id
            if len(chain) > depth:
                target = chain[depth]
            rep[n.id] = target

        g = ModelGraph(self.name, self.framework)
        keep_containers = {a for n in self.nodes.values() for a in self.ancestors(n.id)}
        for n in self.nodes.values():
            if n.is_compute and rep.get(n.id) == n.id:
                g.add_node(_copy_node(n))
            elif n.is_container and n.id in set(rep.values()):
                c = _copy_node(n)
                c.op = "module"
                c.attrs = dict(c.attrs, collapsed=True)
                members = [m for m, r in rep.items() if r == n.id]
                if members and all(self.nodes[m].attrs.get("unused") for m in members):
                    c.attrs["unused"] = True
                c.params = {}
                for m in members:
                    for p in self.nodes[m].params.values():
                        c.params[p.path[len(n.id) + 1:] if n.id else p.path] = p
                ins = [e for e in self.edges if rep.get(e.dst) == n.id and rep.get(e.src) != n.id]
                outs = [e for e in self.edges if rep.get(e.src) == n.id and rep.get(e.dst) != n.id]
                if ins:
                    c.in_shapes = [self.nodes[ins[0].dst].in_shapes[0]] if self.nodes[ins[0].dst].in_shapes else []
                if outs:
                    c.out_shapes = self.nodes[outs[-1].src].out_shapes
                g.add_node(c)
            elif n.is_container and n.id in keep_containers:
                g.add_node(_copy_node(n))
        g.link_children()
        for e in self.edges:
            s, d = rep.get(e.src, e.src), rep.get(e.dst, e.dst)
            if s in g.nodes and d in g.nodes:
                g.add_edge(s, d, e.shape, e.implicit)
        return g

    def subgraph(self, root: str) -> "ModelGraph":
        """Nodes inside ``root``'s subtree plus boundary input/output stubs."""
        if root == "":
            return self
        g = ModelGraph(f"{self.name}.{root}", self.framework)
        inside = {n.id for n in self.nodes.values() if self.in_subtree(n.id, root)}
        for n in self.nodes.values():
            if n.id in inside:
                c = _copy_node(n)
                if n.id == root:
                    c.parent = None
                g.add_node(c)
        g.link_children()
        in_stub, out_stub = f"@in:{root}", f"@out:{root}"
        for e in self.edges:
            si, di = e.src in inside, e.dst in inside
            if si and di:
                g.add_edge(e.src, e.dst, e.shape, e.implicit)
            elif di and not si:
                if in_stub not in g.nodes:
                    g.add_node(Node(in_stub, "Input", "Input", "input", parent=root,
                                    out_shapes=[e.shape] if e.shape else [], label="z"))
                g.add_edge(in_stub, e.dst, e.shape)
            elif si and not di:
                if out_stub not in g.nodes:
                    g.add_node(Node(out_stub, "Output", "Output", "output", parent=root,
                                    in_shapes=[e.shape] if e.shape else [], label="out"))
                g.add_edge(e.src, out_stub, e.shape)
        g.link_children()
        return g

    # ----------------------------------------------------------------- export
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "framework": self.framework,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [{"src": e.src, "dst": e.dst, "shape": list(e.shape) if e.shape else None,
                       "implicit": e.implicit} for e in self.edges],
        }

    def summary(self, max_rows: int = 200) -> str:
        rows = [("path", "kind", "output shape", "#params")]
        total = 0
        seen: set[str] = set()

        def walk(nid: str, indent: int):
            nonlocal total
            n = self.nodes[nid]
            if n.op in ("input", "output"):
                return
            base = (n.label or n.id.split(".")[-1]) if n.op in ("module", "container") else n.id.split(".")[-1]
            name = ("  " * indent) + (base if n.id else self.name)
            shape = n.out_shapes[0] if n.out_shapes else ""
            own = n.n_params
            fresh = [p for p in n.params.values() if p.trainable and p.path not in seen]
            seen.update(p.path for p in fresh)
            total += sum(p.numel for p in fresh)  # tied / shared parameters count once
            np_str = f"{own:,}" if n.is_compute and own else ""
            if n.is_container:
                sub = sum(self.nodes[d].n_params for d in self.descendants(nid) if self.nodes[d].is_compute)
                np_str = f"({sub:,})"
            rows.append((name, n.kind if n.is_compute else n.type_name, str(tuple(shape)) if shape else "", np_str))
            for c in n.children:
                walk(c, indent + 1)

        if "" in self.nodes:
            walk("", 0)
        else:
            for n in self.compute_nodes():
                walk(n.id, 0)
        rows = rows[: max_rows + 1]
        widths = [max(len(r[i]) for r in rows) for i in range(4)]
        line = "─" * (sum(widths) + 9)
        out = [line, " │ ".join(r.ljust(w) for r, w in zip(rows[0], widths)), line]
        out += [" │ ".join(r.ljust(w) for r, w in zip(row, widths)) for row in rows[1:]]
        out += [line, f"total parameters: {total:,}", line]
        return "\n".join(out)

    def __repr__(self) -> str:
        return (f"ModelGraph({self.name!r}, framework={self.framework!r}, "
                f"nodes={len(self.nodes)}, edges={len(self.edges)})")


def _copy_node(n: Node) -> Node:
    return Node(
        id=n.id, kind=n.kind, type_name=n.type_name, op=n.op, parent=n.parent, children=[],
        params=dict(n.params), roles=dict(n.roles), attrs=dict(n.attrs), in_shapes=list(n.in_shapes),
        out_shapes=list(n.out_shapes), module=n.module, label=n.label,
    )


def iter_shapes(obj) -> Iterable[tuple]:
    """Yield shapes of array-like leaves in (nested) tuples/lists/dicts."""
    if hasattr(obj, "shape") and not isinstance(obj, type):
        yield tuple(int(s) for s in obj.shape)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            yield from iter_shapes(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from iter_shapes(o)
