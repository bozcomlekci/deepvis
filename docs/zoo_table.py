"""Write the compatibility table (docs/zoo.md) and its one-line summary (README.md) from the model zoo's results.

    DEEPVIS_ZOO_REPORT=1 pytest -m zoo     # records docs/compat.json
    python docs/zoo_table.py               # rewrites the text between the zoo markers in both files
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
START, END = "<!-- zoo:start -->", "<!-- zoo:end -->"
SUMMARY = "<!-- zoo-summary:start -->", "<!-- zoo-summary:end -->"


def _params(n) -> str:
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K" if n >= 1e3 else str(n)


def table(results: dict) -> str:
    rows = ["| model | from | framework | layers drawn | parameters | page | |", "|---|---|---|---:|---:|---:|---|"]
    for name, r in results.items():
        ok = r.get("status") == "ok"
        rows.append(f"| {name} | {r['source']} | {r['framework']} | {r.get('nodes', '—')} | "
                    f"{_params(r['params']) if 'params' in r else '—'} | "
                    f"{str(r['page_mb']) + ' MB' if 'page_mb' in r else '—'} | {'✓' if ok else r.get('status', '?')} |")
    n_ok = sum(r.get("status") == "ok" for r in results.values())
    note = (f"{n_ok} of {len(results)} architectures wrap, trace to the native model's output (to 1e-4) and make a page"
            " — built from random weights or tiny configs, so nothing is downloaded. Regenerate with "
            "`DEEPVIS_ZOO_REPORT=1 pytest -m zoo && python docs/zoo_table.py`.")
    return "\n".join([note, "", *rows])


def summary(results: dict) -> str:
    n_ok = sum(r.get("status") == "ok" for r in results.values())
    sources = sorted({r["source"] for r in results.values()})
    return (f"{n_ok} of {len(results)} architectures — from {', '.join(sources)} — wrap, trace to the native model's "
            "output (to 1e-4) and make a page in the test suite: [the full table](docs/zoo.md).")


def _replace(path: str, markers: tuple[str, str], body: str) -> None:
    text = open(path, encoding="utf-8").read()
    start, end = markers
    if start not in text or end not in text:
        raise SystemExit(f"{path} has no {start} … {end} markers")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + start + "\n" + body + "\n" + end + tail)


def main() -> None:
    results = json.load(open(os.path.join(HERE, "compat.json"), encoding="utf-8"))
    _replace(os.path.join(HERE, "zoo.md"), (START, END), table(results))
    _replace(os.path.join(HERE, "..", "README.md"), SUMMARY, summary(results))
    print(f"zoo table: {len(results)} models")


if __name__ == "__main__":
    main()
