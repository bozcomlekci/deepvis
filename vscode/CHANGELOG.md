# Changelog

## 0.3.0
- First public release, on the VS Code Marketplace and Open VSX (`batuoz.deepvis`).
- Pages in the panel work offline: the math typesetting (KaTeX) is bundled, and the panel loads nothing from the
  network.
- An installed `deepvis` is used only if it is at least as new as the bundled one; an older install is skipped
  (scripts) or explained (notebook kernels) instead of failing.
- Declares Workspace Trust: deepvis runs your code, so it is off in Restricted Mode and in virtual workspaces.
- Keras models and saved model files (ONNX, `.keras`, `torch.export`, state_dicts, `.safetensors`) in the
  description and docs.

## 0.2.0
- Notebooks: CodeLens and "Visualize model here…" work in Jupyter cells and use the kernel's live objects
  (nothing re-run); "Visualize a live model…" in the notebook toolbar lists the kernel's models and tensors.
- Live pages (`dv.visualize(..., live=True)`, `dv.watch`, `net.serve`) open in the panel, from the extension's
  runs, the integrated terminal or a notebook kernel.
- "Compare with checkpoint…" in the panel's toolbar.
- Downloads and links inside live pages work in the panel.

## 0.1.0
- Visualize a `dv.visualize(...)` call, a model-creating line, or any model expression at the cursor, in a
  panel beside the code; re-made on save; explained failures with a squiggle on the failing line.
