"""deepvis.capture: run a script, capture one visualize call (or a model picked by line), explain failures."""

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.torch

SCRIPT = textwrap.dedent('''\
    import torch
    import torch.nn as nn
    import deepvis as dv


    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            h = self.fc(x)
            return torch.tanh(h)


    def main():
        net = Net()
        x = torch.randn(3, 4)
        dv.visualize(net, x, title="first")
        dv.visualize(net, x,
                     title="second")
        raise RuntimeError("the rest of the script must not run")


    net = Net()
    x = torch.randn(2, 4)
    if __name__ == "__main__":
        main()
    ''')


def capture(tmp_path, *args, code=SCRIPT, stdin=False):
    script = tmp_path / "script.py"
    if not stdin:
        script.write_text(code, encoding="utf-8")
    out = tmp_path / "page.html"
    cmd = [sys.executable, "-m", "deepvis.capture", str(out), str(script), *args] + (["--source", "-"] if stdin else [])
    proc = subprocess.run(cmd, input=code if stdin else None, capture_output=True, text=True, timeout=240,
                          env={**os.environ, "DEEPVIS_CAPTURE": "", "PYTHONPATH": os.pathsep.join(sys.path)})
    result = json.loads((tmp_path / "page.html.json").read_text(encoding="utf-8"))
    assert (proc.returncode == 0) == result["ok"], proc.stderr
    if result["ok"]:
        result["title"] = out.read_text(encoding="utf-8").split("<title>", 1)[1].split(" · ", 1)[0]
    else:
        assert "--- deepvis ---" in proc.stderr
    return result


def test_first_call_and_stop(tmp_path):
    r = capture(tmp_path)
    assert r["ok"] and r["title"] == "first"  # and the RuntimeError after the calls never ran


def test_chosen_call_by_lines(tmp_path):
    assert capture(tmp_path, "--lines", "20-21")["title"] == "second"


def test_model_at_a_line(tmp_path):
    assert capture(tmp_path, "--line", "26", "--model", "net", "--inputs", "x", "--title", "picked")["title"] == "picked"


def test_model_mode_skips_the_scripts_own_calls(tmp_path):
    r = capture(tmp_path, "--line", "22", "--model", "net", "--inputs", "x[:1]", "--title", "mine")
    assert not r["ok"]  # line 22 raises: the calls on lines 19-21 must not have produced the page instead
    code = SCRIPT.replace('    raise RuntimeError("the rest of the script must not run")\n', "    y = 1\n")
    assert capture(tmp_path, "--line", "22", "--model", "net", "--inputs", "x[:1]", "--title", "mine",
                   code=code)["title"] == "mine"


def test_model_in_a_function_and_a_method(tmp_path):
    assert capture(tmp_path, "--line", "18", "--model", "net", "--inputs", "x")["ok"]
    assert capture(tmp_path, "--line", "12", "--model", "self", "--inputs", "x")["ok"]  # inside forward()


def test_name_not_defined_yet(tmp_path):
    r = capture(tmp_path, "--line", "25", "--model", "net", "--inputs", "x")
    assert not r["ok"] and "x" in r["message"] and "fix" in r["message"] and r["line"] == 25


def test_line_that_never_runs(tmp_path):
    code = SCRIPT.replace('if __name__ == "__main__":\n    main()', 'if False:\n    main()')
    r = capture(tmp_path, "--line", "18", "--model", "net", "--inputs", "x", code=code)
    assert not r["ok"] and "never ran" in r["title"]


def test_syntax_error_from_stdin(tmp_path):
    r = capture(tmp_path, code="import deepvis as dv\nx = (1,\n", stdin=True)
    assert not r["ok"] and r["line"] == 2 and "x = (1," in r["message"]


def test_missing_module(tmp_path):
    r = capture(tmp_path, code="import not_a_real_package_xyz\n")
    assert not r["ok"] and "not_a_real_package_xyz" in r["message"] and "pip install" in r["message"]


def test_model_error_is_explained_with_its_line(tmp_path):
    code = SCRIPT.replace("x = torch.randn(3, 4)", "x = torch.randn(3, 5)")
    r = capture(tmp_path, code=code)
    assert not r["ok"] and r["line"] == 19 and "expects 4 features" in r["message"]


def test_no_call(tmp_path):
    r = capture(tmp_path, code="x = 1\n")
    assert not r["ok"] and "no dv.visualize" in r["message"]


def test_sys_exit_before_page(tmp_path):
    r = capture(tmp_path, code="import sys\nsys.exit(3)\n")
    assert not r["ok"] and "exited" in r["title"]
