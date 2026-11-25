# deepvis for VS Code

See your network beside your code. Point at a PyTorch, JAX or Keras model (or one your code loads from ONNX, `.keras` or `.safetensors`), and its
interactive [deepvis](https://batuoz.com/deepvis/docs/readme.html) page opens in a panel next to the editor. The page shows every layer drawn
from its own numbers, the inputs, the intermediate tensors, the outputs, attention and the math.

![deepvis in VS Code: Visualize, layers explained, the forward pass, 3-D tensors, neurons, weight edits, interventions, equations, drawing an input, live training](https://batuoz.com/deepvis/media/deepvis-teaser-small.gif)

<sub>A 60-second tour ([full-size MP4](https://batuoz.com/deepvis/media/deepvis-teaser.mp4)) · [the gallery](https://batuoz.com/deepvis/) · [the tour of
every feature](https://batuoz.com/deepvis/docs/tour.html)</sub>

## Three ways in

| Where | What |
|---|---|
| **`Visualize`** above a `dv.visualize(model, x)` call | runs the file up to that call and shows its page |
| **`Visualize net…`** above a line that makes a model (`net = MyNet()`, `nn.Sequential(…)`, `from_pretrained(…)`) | asks for the inputs (suggesting tensors from the file, e.g. `x`, or type `(x, t)`) and shows the page |
| Right-click a name → **deepvis: Visualize model here…** (`⌘⌥V` / `Ctrl+Alt+V`) | the same, for any model expression at that point in the code (also inside functions and `forward`) |

The **graph** button in the editor title visualizes the file: its `dv.visualize` call, or its model.

## Notebooks

In a Jupyter notebook the objects already live in the kernel, trained weights and all. deepvis evaluates the model
and inputs there, so nothing is re-run:

* the same **Visualize** links appear in code cells;
* **deepvis: Visualize a live model…** in the notebook toolbar lists the kernel's models and tensors, with their
  shapes.

The first time, VS Code asks you to allow deepvis to use the kernel (that's the Jupyter extension's permission
prompt).

## Training, live

A live page opens in the panel instead of a browser tab and updates as the model trains. It can come from a script
run by the extension, from the integrated terminal, or from a notebook:

```python
srv = dv.visualize(model, x, live=True)       # or: with dv.watch(model, x) as w: … w.step(step, loss=loss)
for step in range(1000):
    loss = train_step()
    srv.update(step, loss=loss)
```

The Monitor tab's replay slider steps back through the run. **Stop** in the panel ends a run the extension started.

## Compare

**Compare with checkpoint…** (panel toolbar) puts the current model next to a `.pt` / `.pth` / `.safetensors`
checkpoint of the same architecture. The Compare tab shows how far every layer's weights and activations moved,
and both outputs.

## While you work

- **Your unsaved edits run.** The editor's current text is what runs.
- **Saving the file re-makes the page** (`deepvis.refreshOnSave`).
- **The run stops as soon as the page is made,** so training code after it doesn't run.
- **A failure shows what went wrong and how to fix it,** with a red squiggle on the failing line and *Go to line*.
- **Panel toolbar:** ↻ run again, save the page as a standalone `.html`, or open it in the browser.
- **Downloads work in the panel.** The page's own downloads (⤓ SVG, edits) open a save dialog.

## Requirements

* **Python 3.10 or newer** with your model's framework installed (torch, jax / flax / equinox, keras, onnxruntime …)
  and numpy. It runs with the interpreter selected in the Python extension, or `deepvis.pythonPath`.
* **deepvis itself is bundled.** If that interpreter has its own `pip install deepvis` that is at least as new, that
  copy is used instead; an older one is skipped (the Output channel says which ran). In a notebook, upgrade the
  kernel's copy with `%pip install -U deepvis` if the extension asks you to.
* For notebooks, the Jupyter extension.

## Workspace Trust

deepvis runs your Python code to draw the model, so it only works in a
[trusted workspace](https://code.visualstudio.com/docs/editor/workspace-trust): in Restricted Mode the extension
stays disabled until you trust the folder.

## Privacy

deepvis collects no telemetry. Your code runs locally, in your interpreter, and the pages it makes stay on your
machine. Pages shown in the panel make no network requests: the math typesetting (KaTeX) is bundled.

## Settings

| Setting | Default | |
|---|---|---|
| `deepvis.pythonPath` | "" | the Python to run with ("" = the Python extension's choice) |
| `deepvis.refreshOnSave` | true | re-make the page on save |
| `deepvis.timeoutSeconds` | 300 | stop runs that take longer |
| `deepvis.codeLens` | true | show the Visualize links |
| `deepvis.runInFileDir` | false | run in the file's folder instead of the workspace folder |

## Develop

```bash
npm install
npm run vendor                                               # bundle ../src/deepvis into python/vendor
DEEPVIS_TEST_PYTHON=/path/to/python npm test                 # type check, parser and run.py tests, then integration
                                                             # tests in a real VS Code
npm run package                                              # → deepvis-<version>.vsix
code --install-extension deepvis-0.3.0.vsix
```

Releases are published by CI from a `vscode-v<version>` tag.
