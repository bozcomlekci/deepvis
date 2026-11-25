"""The model zoo: every architecture wraps, traces to the native output, and makes a page — the evidence behind the
compatibility table in the README (``DEEPVIS_ZOO_REPORT=1 pytest -m zoo && python docs/zoo_table.py``)."""

import json
import os
import time

import numpy as np
import pytest

import deepvis as dv
from deepvis.core.tensor import first_array, to_numpy
from zoo_models import ZOO

RESULTS: dict = {}
REPORT = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "compat.json")


def _params():
    for e in ZOO:
        marks = [pytest.mark.zoo] + ([] if e.fast else [pytest.mark.zoo_slow])
        if e.framework in ("flax", "nnx", "equinox"):
            marks.append(pytest.mark.jax)
        if e.framework == "keras":
            marks.append(pytest.mark.keras)
        yield pytest.param(e, id=e.id, marks=marks)


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    if RESULTS and os.environ.get("DEEPVIS_ZOO_REPORT"):
        old = json.load(open(REPORT, encoding="utf-8")) if os.path.exists(REPORT) else {}
        old.update(RESULTS)
        with open(REPORT, "w", encoding="utf-8") as f:
            json.dump(dict(sorted(old.items(), key=lambda kv: (kv[1]["source"], kv[0]))), f, indent=1)


@pytest.mark.parametrize("entry", list(_params()))
def test_zoo(entry, tmp_path):
    if not entry.available():
        pytest.skip(f"needs {', '.join(entry.needs)}")
    framework = entry.framework
    if framework == "keras":  # which backend ran it
        framework = f"keras · {os.environ.get('KERAS_BACKEND', '?')}"
    rec = {"source": entry.source, "framework": framework}
    RESULTS[entry.id] = rec
    try:
        model, inputs, native = entry.build()
        want = to_numpy(first_array(native()))
        t0 = time.perf_counter()
        net = dv.wrap(model, sample_input=inputs)
        tr = net.trace(*inputs)
        rec["trace_s"] = round(time.perf_counter() - t0, 2)
        got = to_numpy(first_array(tr.output))
        assert got.shape == want.shape, f"output shape {got.shape} vs native {want.shape}"
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5, err_msg="output differs from the native model")
        compute = [n for n in net.graph.compute_nodes() if n.op in ("module", "function")]
        rec.update(nodes=len(compute), params=net.num_params(), trace_mode=getattr(net.backend, "trace_mode", "jax"))
        assert len(compute) >= 2, "drawn as one opaque box"
        t1 = time.perf_counter()
        page = net.export_html(str(tmp_path / "p.html"), *inputs)
        rec["page_s"] = round(time.perf_counter() - t1, 2)
        rec["page_mb"] = round(os.path.getsize(page) / 1e6, 2)
        assert rec["page_mb"] < 16
        rec["status"] = "ok"
    except BaseException as exc:
        rec["status"] = f"fails: {type(exc).__name__}: {str(exc).strip().splitlines()[0][:120] if str(exc).strip() else ''}"
        raise
