"""Run a Python script and capture the page one of its ``dv.visualize(...)`` calls makes — or visualize any model /
inputs expressions at a chosen line. Used by the VS Code extension and the demo app.

    python -m deepvis.capture OUT.html SCRIPT.py                      # the first dv.visualize(...) call
    python -m deepvis.capture OUT.html SCRIPT.py --lines 12-14        # the call written on lines 12-14
    python -m deepvis.capture OUT.html SCRIPT.py --line 9 --model net --inputs "x, t"
                                                                      # dv.visualize(net, (x, t)) right after line 9

``--source -`` reads the code from stdin instead of SCRIPT.py (an unsaved editor buffer); SCRIPT.py still names it
in tracebacks. The script runs as ``__main__`` with its own directory on ``sys.path``, like ``python SCRIPT.py``,
and stops as soon as the page is written. The outcome goes to ``OUT.html.json``::

    {"ok": true,  "page": "OUT.html", "seconds": 1.2}
    {"ok": false, "title": "…", "message": "what went wrong\\n  → fix: …", "file": "SCRIPT.py", "line": 12}

and, on failure, the message is also printed to stderr after a ``--- deepvis ---`` marker.
"""

from __future__ import annotations

import argparse
import ast
import json
import linecache
import os
import sys
import time
import traceback

from .errors import Captured  # noqa: F401 - re-exported: raised by visualize() once the page is written

MARK = "--- deepvis ---"
# The interface the VS Code extension relies on (arguments, the OUT.html.json result, live-page announcements).
# Raise it when that changes, so the extension uses its bundled copy rather than an older install.
PROTOCOL = 1


class Failure(Exception):
    def __init__(self, title: str, message: str, line: int | None = None):
        super().__init__(message)
        self.title, self.message, self.line = title, message, line


# ------------------------------------------------------------------------------------------ explanations
def _script_line(exc: BaseException, script: str) -> int | None:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if os.path.abspath(f.filename) == script]
    return frames[-1].lineno if frames else None


def explain(exc: BaseException, code: str, script: str) -> tuple[str, int | None]:
    """A short explanation of why the script failed (what, where, how to fix) and the script line it failed on."""
    from .errors import DeepvisError
    from .errors import explain as hint_for

    lines = code.splitlines()
    if isinstance(exc, SyntaxError):
        n = exc.lineno or 0
        src = lines[n - 1] if 0 < n <= len(lines) else (exc.text or "").rstrip("\n")
        caret = " " * max((exc.offset or 1) - 1, 0) + "^"
        return (f"line {exc.lineno}: {exc.msg}\n    {src.strip()}\n    {caret[len(src) - len(src.lstrip()):]}\n"
                "  → fix: correct the Python on that line (a missing bracket, colon or quote is the usual cause)."
                ), exc.lineno
    if isinstance(exc, ModuleNotFoundError):
        have = os.environ.get("DEEPVIS_AVAILABLE")
        where = f"this demo (it has {have})" if have else f"this Python ({sys.executable})"
        fix = (f"use one of those, or run it on your machine: pip install {exc.name}" if have
               else f"install it: {sys.executable} -m pip install {exc.name} — or pick the interpreter that has it.")
        return f"the package {exc.name!r} isn't installed in {where}.\n  → fix: {fix}", _script_line(exc, script)
    where = _script_line(exc, script)
    at = f"line {where}: " if where else ""
    code_line = f"\n    {lines[where - 1].strip()}" if where and where <= len(lines) else ""
    if isinstance(exc, DeepvisError):
        return f"{at}{exc.what}{code_line}" + (f"\n  → fix: {exc.fix}" if exc.fix else ""), where
    first = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    hint = hint_for(exc)
    if "see the traceback above" in hint or hint.startswith("check that the model runs"):
        hint = "fix the code on the line shown."
    return f"{at}{type(exc).__name__}: {first}{code_line}\n  → fix: {hint}", where


# ------------------------------------------------------------------------------------------ --model mode
def _scope_of(tree: ast.Module, line: int):
    """(statement containing ``line``, the def enclosing it or None) — descending into if/for/with/try blocks."""
    body, func = tree.body, None
    while True:
        stmt = next((s for s in body if s.lineno <= line <= (s.end_lineno or s.lineno)), None)
        if stmt is None:
            return None, func
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func, body = stmt, stmt.body
            continue
        if isinstance(stmt, ast.ClassDef):
            if any(s.lineno <= line <= (s.end_lineno or s.lineno) for s in stmt.body):
                body = stmt.body  # a method's body is its own scope; a class attribute runs with the class
                if not any(isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef)) and
                           s.lineno <= line <= (s.end_lineno or s.lineno) for s in body):
                    return stmt, func
                continue
            return stmt, func
        flat = [b for f in ("body", "orelse", "finalbody") for b in (getattr(stmt, f, None) or [])]
        for h in (getattr(stmt, "handlers", None) or []) + (getattr(stmt, "cases", None) or []):
            flat.extend(h.body)
        nxt = [s for s in flat if s.lineno <= line <= (s.end_lineno or s.lineno)]
        if not nxt:
            return stmt, func
        body = flat


class _AtLine:
    """A tracer that calls ``fn(frame)`` right after the statement on ``line`` has run (in its own scope)."""

    def __init__(self, script: str, code: str, line: int, fn):
        tree = ast.parse(code)
        self.stmt, func = _scope_of(tree, line)
        if self.stmt is None:
            raise Failure("nothing to run on that line",
                          f"line {line} has no code.\n  → fix: put the cursor on a line that assigns or uses the model.",
                          line)
        self.first = min([func.lineno] + [d.lineno for d in func.decorator_list]) if func else None
        self.script, self.fn, self.seen, self.raised = script, fn, False, False

    def _mine(self, frame) -> bool:
        c = frame.f_code
        if os.path.abspath(c.co_filename) != self.script:
            return False
        return c.co_name == "<module>" if self.first is None else c.co_firstlineno == self.first

    def __call__(self, frame, event, arg):  # the global tracer: only trace the scope that holds the line
        return self._local if event == "call" and self._mine(frame) else None

    def _local(self, frame, event, arg):
        lo, hi = self.stmt.lineno, self.stmt.end_lineno or self.stmt.lineno
        if event == "line":
            self.raised = False
            if lo <= frame.f_lineno <= hi:
                self.seen = True
            elif self.seen:
                self._fire(frame)
        elif event == "exception":
            self.raised = True  # if the frame returns now, it's the exception leaving: report that instead
        elif event == "return" and self.seen and not self.raised:
            self._fire(frame)
        return self._local

    def _fire(self, frame):
        sys.settrace(None)
        self.fn(frame, self.stmt.end_lineno or self.stmt.lineno)


def _visualize_at(out: str, model_expr: str, inputs_expr: str, params_expr: str | None, title: str | None,
                  options: dict | None = None):
    import deepvis as dv

    def fn(frame, after: int):
        env = dict(frame.f_globals)
        env.update(frame.f_locals)

        def ev(expr, what):
            try:
                return eval(expr, env)  # noqa: S307 - the user's own expression, in the user's own script
            except Exception as exc:
                raise Failure(f"couldn't evaluate the {what}",
                              f"{what} `{expr}` failed right after line {after}: "
                              f"{type(exc).__name__}: {exc}\n  → fix: use a name that exists at that point of the "
                              f"script, or move the cursor below the line that defines it.", after) from exc

        model = ev(model_expr, "model")
        inputs = ev(inputs_expr, "inputs") if inputs_expr.strip() else None
        params = ev(params_expr, "params") if params_expr else None
        kw = dict(options or {}, **({"title": title} if title else {}))
        raise Captured(dv.visualize(model, inputs, params, path=out, **kw).page)

    return fn


# ------------------------------------------------------------------------------------------ main
def run(out: str, script: str, code: str, *, lines: str | None = None, line: int | None = None,
        model: str | None = None, inputs: str = "", params: str | None = None, title: str | None = None,
        options: dict | None = None) -> dict:
    """Run ``code`` (the text of ``script``) and write the captured page to ``out``; returns the outcome dict."""
    from . import CAPTURE_ENV, CAPTURE_LINES_ENV, CAPTURE_OPTIONS_ENV, CAPTURE_STOP_ENV

    script, out = os.path.abspath(script), os.path.abspath(out)
    if options:
        os.environ[CAPTURE_OPTIONS_ENV] = json.dumps(options)
    t0 = time.time()
    if os.path.exists(out):
        os.remove(out)
    os.environ[CAPTURE_ENV] = out
    os.environ[CAPTURE_STOP_ENV] = "1"
    if model:
        os.environ[CAPTURE_LINES_ENV] = f"{script}:0-0"  # the script's own visualize calls make no page
    elif lines:
        os.environ[CAPTURE_LINES_ENV] = f"{script}:{lines}"
    linecache.cache[script] = (len(code), None, code.splitlines(True), script)  # tracebacks show the run text
    sys.argv = [script]
    sys.path.insert(0, os.path.dirname(script))
    g = {"__name__": "__main__", "__file__": script, "__builtins__": __builtins__}
    tracer = None
    try:
        compiled = compile(code, script, "exec")
        if model:
            tracer = _AtLine(script, code, line or 1, _visualize_at(out, model, inputs, params, title, options))
            sys.settrace(tracer)
        exec(compiled, g)  # noqa: S102 - running the user's script is the point
        sys.settrace(None)
        if tracer is not None:
            raise Failure("that line never ran",
                          f"line {line} didn't run when the file was executed (it's inside a function that isn't "
                          "called, or behind an `if` that was false).\n  → fix: put the cursor on a line that runs, "
                          "or call dv.visualize(model, inputs) where the model exists.", line)
    except Captured as c:
        sys.settrace(None)
        return {"ok": True, "page": c.page, "seconds": round(time.time() - t0, 2)}
    except Failure as f:
        sys.settrace(None)
        return {"ok": False, "title": f.title, "message": f.message, "file": script, "line": f.line}
    except SystemExit as e:
        sys.settrace(None)
        if os.path.exists(out):
            return {"ok": True, "page": out, "seconds": round(time.time() - t0, 2)}
        return {"ok": False, "title": "the script exited before making a page",
                "message": f"it called sys.exit({e.code!r}) before dv.visualize(...) ran.\n  → fix: call "
                           "dv.visualize(model, inputs) before the script exits.", "file": script, "line": None}
    except BaseException as exc:  # noqa: BLE001 - explain it, briefly
        sys.settrace(None)
        if isinstance(exc, KeyboardInterrupt):
            return {"ok": False, "title": "stopped", "message": "the run was interrupted.", "file": script, "line": None}
        traceback.print_exc()
        message, where = explain(exc, code, script)
        return {"ok": False, "title": "the script didn't produce a page", "message": message, "file": script,
                "line": where}
    finally:
        sys.settrace(None)
    if os.path.exists(out):
        return {"ok": True, "page": out, "seconds": round(time.time() - t0, 2)}
    live = _live_urls()
    if live:  # dv.visualize(..., live=True): the page is served, not written
        return {"ok": True, "page": None, "live": live[0], "seconds": round(time.time() - t0, 2)}
    which = f" on lines {lines}" if lines else ""
    return {"ok": False, "title": "the script ran but made no page",
            "message": f"the script ran but made no page: no dv.visualize(...) call ran{which}.\n  → fix: end it with  "
                       "dv.visualize(model, inputs)  (import deepvis as dv at the top), or pick a model with "
                       "“Visualize model here…”.", "file": script, "line": None}


def _live_urls() -> list[str]:
    srv = sys.modules.get("deepvis.viz.server")
    return [r.url for r in getattr(srv, "RUNNING", [])] if srv else []


# ------------------------------------------------------------------------------------------ in a live kernel
def _namespace(env: dict | None) -> dict:
    if env is not None:
        return env
    try:
        return get_ipython().user_ns  # type: ignore[name-defined]  # noqa: F821 - defined inside IPython
    except NameError:
        import __main__

        return vars(__main__)


def _outcome(out: str, make) -> dict:
    """Run ``make()`` (which writes the page to ``out``), turning failures into the outcome dict; write OUT.json."""
    t0 = time.time()
    try:
        page = make()
        result = {"ok": True, "page": page, "seconds": round(time.time() - t0, 2)}
        live = _live_urls()
        if page is None and live:
            result.update(live=live[-1])
        elif page is None:
            result = {"ok": False, "title": "no page was made", "message": "the call returned without a page.",
                      "file": None, "line": None}
    except Failure as f:
        result = {"ok": False, "title": f.title, "message": f.message, "file": None, "line": None}
    except Exception as exc:  # noqa: BLE001 - explain it, briefly
        message, _ = explain(exc, "", "<kernel>")
        result = {"ok": False, "title": "couldn't make the page", "message": message, "file": None, "line": None}
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f)
    return result


def visualize_to(out: str, model: str, inputs: str = "", params: str | None = None, title: str | None = None,
                 env: dict | None = None, **options) -> dict:
    """In a running interpreter (a Jupyter kernel): evaluate the ``model`` / ``inputs`` expressions in its namespace
    and write their page to ``out`` — the objects as they are now (trained weights and all), nothing re-run.
    Writes the outcome to ``OUT.json`` like :func:`run` and returns it."""
    import deepvis as dv

    out = os.path.abspath(out)
    ns = _namespace(env)

    def ev(expr, what):
        try:
            return eval(expr, ns)  # noqa: S307 - the user's own expression, in the user's own kernel
        except Exception as exc:
            raise Failure(f"couldn't evaluate the {what}",
                          f"{what} `{expr}` failed in the kernel: {type(exc).__name__}: {exc}\n  → fix: run the "
                          "cell that defines it first, or pick another name.") from exc

    def make():
        m = ev(model, "model")
        x = ev(inputs, "inputs") if inputs.strip() else None
        p = ev(params, "params") if params else None
        kw = dict(options, **({"title": title} if title else {}))
        return dv.visualize(m, x, p, path=out, **kw).page

    return _outcome(out, make)


def kernel_call(out: str, source: str, env: dict | None = None, **options) -> dict:
    """In a running interpreter: evaluate one ``dv.visualize(...)`` call (its source text, from a notebook cell)
    with its page going to ``out`` — on the live objects, without re-running the cell."""
    from . import CAPTURE_ENV, CAPTURE_LINES_ENV, CAPTURE_OPTIONS_ENV, CAPTURE_STOP_ENV

    out = os.path.abspath(out)
    ns = _namespace(env)
    saved = {k: os.environ.pop(k, None) for k in (CAPTURE_ENV, CAPTURE_LINES_ENV, CAPTURE_STOP_ENV, CAPTURE_OPTIONS_ENV)}

    def make():
        os.environ[CAPTURE_ENV] = out
        if options:
            os.environ[CAPTURE_OPTIONS_ENV] = json.dumps(options)
        try:
            code = compile(source.strip(), "<cell>", "eval")
        except SyntaxError as exc:
            raise Failure("that call isn't valid Python", f"{exc.msg}: {source.strip()[:200]}") from exc
        try:
            r = eval(code, ns)  # noqa: S307 - the user's own call, in the user's own kernel
        except NameError as exc:
            raise Failure("a name in the call isn't defined in the kernel",
                          f"{exc}\n  → fix: run the cells above it first (the kernel doesn't know it yet).") from exc
        page = getattr(r, "page", None)
        return page if page and os.path.exists(page) else (out if os.path.exists(out) else None)

    try:
        return _outcome(out, make)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _describe(name: str, v) -> dict | None:
    """What a kernel variable is, if it's a model or a model input."""
    mod = type(v).__module__ or ""
    shape = getattr(v, "shape", None)
    try:
        if type(v).__name__ == "Net" and mod.startswith("deepvis"):
            return {"name": name, "kind": "model", "type": f"deepvis {v.name}", "params": int(v.num_params())}
        if mod.startswith(("torch", "numpy", "jax", "jaxlib")) and shape is not None and not callable(v):
            return {"name": name, "kind": "input", "type": type(v).__name__, "shape": [int(s) for s in shape],
                    "dtype": str(getattr(v, "dtype", ""))}
        import sys as _sys

        torch = _sys.modules.get("torch")
        if torch is not None and isinstance(v, torch.nn.Module):
            n = sum(p.numel() for p in v.parameters())
            return {"name": name, "kind": "model", "type": type(v).__name__, "params": int(n)}
        if mod.startswith(("flax", "equinox")) or any(c.__module__.startswith(("flax", "equinox"))
                                                      for c in type(v).__mro__[1:] if c.__module__):
            if not isinstance(v, type):
                return {"name": name, "kind": "model", "type": type(v).__name__, "params": None}
        if isinstance(v, dict) and "params" in v and not name.startswith("_"):
            return {"name": name, "kind": "params", "type": "variables", "params": None}
        if isinstance(v, tuple) and 1 < len(v) <= 4 and all(getattr(a, "shape", None) is not None for a in v):
            return {"name": name, "kind": "input", "type": "tuple",
                    "shape": [[int(s) for s in a.shape] for a in v], "dtype": ""}
    except Exception:  # noqa: BLE001 - an odd object: just don't list it
        return None
    return None


def kernel_objects(env: dict | None = None) -> list[dict]:
    """The models and model inputs defined in a running interpreter, biggest models first (the extension's
    "Visualize a live model…" lists these)."""
    ns = _namespace(env)
    found = [d for k, v in list(ns.items()) if not k.startswith("_") and (d := _describe(k, v))]
    found.sort(key=lambda d: (d["kind"] != "model", -(d.get("params") or 0)))
    return found


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m deepvis.capture", description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="the page to write (.html); the outcome goes to OUT.json")
    ap.add_argument("script", help="the Python script to run")
    ap.add_argument("--source", help="read the code from this file instead ('-' = stdin)")
    ap.add_argument("--lines", help="FIRST-LAST: capture only the dv.visualize call written on these lines")
    ap.add_argument("--line", type=int, help="with --model: visualize right after this line has run")
    ap.add_argument("--model", help="an expression for the model, evaluated at --line")
    ap.add_argument("--inputs", default="", help="an expression for its inputs (a tensor or a tuple)")
    ap.add_argument("--params", help="an expression for the params / variables (flax, pure JAX)")
    ap.add_argument("--title", help="the page title")
    ap.add_argument("--options", help='JSON options for the page, e.g. \'{"compare": "ckpt.pt"}\'')
    a = ap.parse_args(argv)
    src = a.source or a.script
    if src == "-":
        code = sys.stdin.read()
    else:
        with open(src, encoding="utf-8") as f:
            code = f.read()
    try:
        options = json.loads(a.options) if a.options else None
    except json.JSONDecodeError as exc:
        ap.error(f"--options is not JSON: {exc}")
    result = run(a.out, a.script, code, lines=a.lines, line=a.line, model=a.model, inputs=a.inputs,
                 params=a.params, title=a.title, options=options)
    with open(os.path.abspath(a.out) + ".json", "w", encoding="utf-8") as f:
        json.dump(result, f)
    sys.stdout.flush()
    if result.get("live"):  # keep serving the live page (its final state) until the run is stopped
        print(f"deepvis: serving {result['live']} — stop the run to close it", flush=True)
        try:
            while _live_urls():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        os._exit(0)  # noqa: SLF001
    if not result["ok"]:
        print(f"{MARK}\n{result['message']}", file=sys.stderr)
        sys.stderr.flush()
        os._exit(1)  # noqa: SLF001 - don't let leftover threads of the script keep the process alive
    os._exit(0)  # noqa: SLF001


if __name__ == "__main__":
    main()
