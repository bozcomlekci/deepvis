"""The extension's entry point: run a script with deepvis.capture, preferring the deepvis installed in the chosen
interpreter and falling back to the copy bundled with the extension (python/vendor/). Arguments are those of
``python -m deepvis.capture``.

An installed deepvis is used only if it speaks at least the bundled copy's capture ``PROTOCOL``, so an older
``pip install deepvis`` can't break a newer extension."""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "vendor")


def _fail(title: str, message: str) -> None:
    out = os.path.abspath(sys.argv[1]) + ".json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"ok": False, "title": title, "message": message, "file": None, "line": None}, f)
    print(f"--- deepvis ---\n{message}", file=sys.stderr)
    sys.exit(1)


def _bundled_protocol() -> int | None:
    """The bundled copy's PROTOCOL, read from its source (importing it would shadow the installed one)."""
    try:
        with open(os.path.join(VENDOR, "deepvis", "capture.py"), encoding="utf-8") as f:
            m = re.search(r"^PROTOCOL = (\d+)", f.read(), re.M)
        return int(m.group(1)) if m else 0
    except OSError:
        return None  # not vendored (running from a checkout): whatever is installed is it


def _forget_deepvis() -> None:
    for name in [m for m in sys.modules if m == "deepvis" or m.startswith("deepvis.")]:
        del sys.modules[name]


def _load():
    bundled = _bundled_protocol()
    try:
        import deepvis.capture as cap  # the user's own install wins, if it is new enough

        have = getattr(cap, "PROTOCOL", 0)
        if bundled is None or have >= bundled:
            print(f"[deepvis] using deepvis {getattr(sys.modules['deepvis'], '__version__', '?')} from "
                  f"{os.path.dirname(cap.__file__)}", flush=True)
            return cap.main
        print(f"[deepvis] the installed deepvis ({os.path.dirname(cap.__file__)}) is older than this extension "
              f"(capture protocol {have} < {bundled}); using the bundled copy — `pip install -U deepvis` to update it",
              flush=True)
    except ImportError:
        pass
    _forget_deepvis()
    sys.path.insert(0, VENDOR)
    try:
        from deepvis.capture import main
        return main
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        _fail("this Python can't run deepvis",
              f"{sys.executable} is missing {missing!r}, which deepvis needs.\n"
              f"  → fix: pick the interpreter your project uses (Python: Select Interpreter), or install it: "
              f"{sys.executable} -m pip install {missing}")


if __name__ == "__main__":
    if sys.version_info < (3, 10):
        _fail("this Python is too old for deepvis",
              f"{sys.executable} is Python {sys.version.split()[0]}; deepvis needs 3.10 or newer.\n"
              "  → fix: pick a newer interpreter (Python: Select Interpreter) or set deepvis.pythonPath.")
    _load()(sys.argv[1:])
