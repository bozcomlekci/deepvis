// deepvis for VS Code: pick a model (or a dv.visualize call) in the code, see its interactive page beside it.
//
// The script runs in the project's Python (python/run.py → deepvis.capture) with the editor's text, so unsaved
// edits count; the page deepvis writes is shown in a webview panel, re-made on save.
// @ts-check
"use strict";

const vscode = require("vscode");
const cp = require("child_process");
const crypto = require("crypto");
const http = require("http");
const fs = require("fs");
const os = require("os");
const path = require("path");
const P = require("./parse");

const VIEW = "deepvis";
let ctx, panel, output, status, diagnostics, tmpDir;
let job = null;           // the running process: {child, req, timer, stopped}
let lastReq = null;       // the request the panel shows: re-run on save / ↻
let lastResult = null;    // {ok, page, title, message, line, …}
let pageCount = 0;
let announce = null;      // {server, url}: where live pages (dv.visualize(..., live=True), net.serve) announce themselves
let live = null;          // the live page the panel shows: {url, pid, title}
const lensChanged = new vscode.EventEmitter();

// ------------------------------------------------------------------------------------------------ activate
function activate(context) {
  ctx = context;
  output = vscode.window.createOutputChannel("deepvis");
  diagnostics = vscode.languages.createDiagnosticCollection("deepvis");
  status = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  status.command = "deepvis.stop";
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "deepvis-vscode-"));
  const cmd = (id, fn) => context.subscriptions.push(vscode.commands.registerCommand(id, fn));
  cmd("deepvis.visualizeFile", visualizeFile);
  cmd("deepvis.visualizeCall", (uri, index) => run({ kind: "call", uri: uri || activeUri(), index: index || 0 }));
  cmd("deepvis.visualizeModel", visualizeModel);
  cmd("deepvis.visualizeLive", visualizeLive);
  cmd("deepvis.rerun", () => lastReq ? run(lastReq) : visualizeFile());
  cmd("deepvis.stop", stop);
  cmd("deepvis.compareCheckpoint", compareCheckpoint);
  cmd("deepvis.savePage", savePage);
  cmd("deepvis.openInBrowser", () => lastResult && lastResult.ok && vscode.env.openExternal(vscode.Uri.file(lastResult.page)));
  context.subscriptions.push(
    output, diagnostics, status, lensChanged,
    vscode.languages.registerCodeLensProvider({ language: "python" }, { provideCodeLenses, onDidChangeCodeLenses: lensChanged.event }),
    vscode.workspace.onDidChangeConfiguration((e) => e.affectsConfiguration("deepvis") && lensChanged.fire()),
    vscode.workspace.onDidSaveTextDocument((doc) => {
      if (panel && lastReq && doc.uri.toString() === lastReq.uri.toString() && config(doc.uri).get("refreshOnSave")) run(lastReq);
    }),
  );
  const ready = startAnnounce(context).catch((e) => output.appendLine(`[deepvis] live pages won't open here: ${e}`));
  return { run, ready, lastResult: () => lastResult, panel: () => panel, live: () => live, announceUrl: () => announce && announce.url };  // for tests
}

function deactivate() {
  if (job) kill(job.child);
  if (announce) announce.server.close();
  try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch (e) { /* already gone */ }
}

const config = (uri) => vscode.workspace.getConfiguration("deepvis", uri);
const activeUri = () => vscode.window.activeTextEditor && vscode.window.activeTextEditor.document.uri;

// ------------------------------------------------------------------------------------------------ live pages
// A live page (LiveServer) started in a terminal, a notebook kernel or a run of ours POSTs {url, title, pid} here —
// its address reaches Python as $DEEPVIS_VSCODE — and the panel then frames it instead of a browser tab opening.
function startAnnounce(context) {
  const token = crypto.randomBytes(12).toString("hex");
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      if (req.method !== "POST" || req.url !== `/${token}/announce`) { res.writeHead(404); return res.end(); }
      let body = "";
      req.on("data", (d) => { body += d; if (body.length > 1e5) req.destroy(); });
      req.on("end", async () => {
        let ev = null;
        try { ev = JSON.parse(body); } catch (e) { /* ignore */ }
        const ok = ev && typeof ev.url === "string" && /^http:\/\/(127\.0\.0\.1|localhost|\[::1\]):\d+\/?$/.test(ev.url);
        if (!ok) { res.writeHead(400, { "Content-Type": "application/json" }); return res.end('{"error":"bad announce"}'); }
        // where the panel will load the page from (Codespaces / tunnels forward it to another host): the live
        // server only answers requests addressed to loopback or to this host
        let shown = null;
        if (!ev.closed) {
          try { shown = await vscode.env.asExternalUri(vscode.Uri.parse(ev.url)); } catch (e) { /* show it as is */ }
        }
        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify(shown ? { host: shown.authority } : {}));
        onAnnounce(ev, shown);
      });
    });
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const url = `http://127.0.0.1:${/** @type {import('net').AddressInfo} */ (server.address()).port}/${token}`;
      announce = { server, url };
      const env = context.environmentVariableCollection;
      env.description = "Lets deepvis live pages started in this terminal open beside your code.";
      env.replace("DEEPVIS_VSCODE", url);
      resolve(url);
    });
  });
}

async function onAnnounce(ev, shown) {
  if (ev.closed) {
    if (live && live.url === ev.url) {
      live = null;
      post({ type: "notice", text: "the live run ended — this is its last state" });
    }
    return;
  }
  shown = shown || await vscode.env.asExternalUri(vscode.Uri.parse(ev.url));
  live = { url: ev.url, pid: ev.pid, title: ev.title || "live" };
  showPanel();
  panel.title = `deepvis · live · ${live.title}`;
  post({ type: "live", url: shown.toString(true) });
  const result = { ok: true, live: ev.url, title: live.title, page: null };
  if (job && job.child.pid === ev.pid) {  // one of our runs went live: keep it running, without the time limit
    job.live = true;
    clearTimeout(job.timer);
    status.text = `$(broadcast) deepvis live: ${live.title}`;
    status.tooltip = "a live deepvis page is running — click to stop it";
    status.show();
    diagnostics.delete(job.req.uri);
    lastResult = result;
    job.resolve(result);
  } else {
    lastResult = result;
  }
}

// ------------------------------------------------------------------------------------------------ code lenses
/** The text whose imports apply to `doc`: the whole notebook for a cell, else the document itself. */
function contextOf(doc) {
  if (doc.uri.scheme !== "vscode-notebook-cell") return doc.getText();
  const nb = notebookOf(doc.uri);
  return nb ? nb.getCells().map((c) => c.document.getText()).join("\n") : "import deepvis as dv";
}

function provideCodeLenses(doc) {
  if (!config(doc.uri).get("codeLens")) return [];
  const text = doc.getText(), lenses = [];
  const at = (line) => new vscode.Range(line - 1, 0, line - 1, 0);
  P.visualizeCalls(text, contextOf(doc)).forEach((c, index) => lenses.push(new vscode.CodeLens(at(c.first), {
    title: "$(graph) Visualize", tooltip: "Run this file up to this call and show the page beside the code",
    command: "deepvis.visualizeCall", arguments: [doc.uri, index],
  })));
  for (const m of P.modelLines(text)) lenses.push(new vscode.CodeLens(at(m.line), {
    title: `$(graph) Visualize ${m.name}…`, tooltip: `Pick inputs for ${m.name} and show its page beside the code`,
    command: "deepvis.visualizeModel", arguments: [doc.uri, m.name, m.line],
  }));
  return lenses;
}

// ------------------------------------------------------------------------------------------------ commands
async function visualizeFile(uri) {
  uri = uri instanceof vscode.Uri ? uri : activeUri();
  if (!uri) return vscode.window.showInformationMessage("deepvis: open a Python file first.");
  const doc = await vscode.workspace.openTextDocument(uri);
  const text = doc.getText();
  const calls = P.visualizeCalls(text);
  if (calls.length === 1) return run({ kind: "call", uri, index: 0 });
  if (calls.length > 1) {
    const pick = await vscode.window.showQuickPick(calls.map((c, index) => ({
      label: `$(graph) line ${c.first}`, description: c.text.length > 80 ? c.text.slice(0, 77) + "…" : c.text, index,
    })), { placeHolder: "Which dv.visualize(...) call?" });
    return pick && run({ kind: "call", uri, index: pick.index });
  }
  const models = P.modelLines(text);
  if (!models.length) {
    return vscode.window.showInformationMessage(
      "deepvis: no dv.visualize(...) call or model found in this file. Select the model's name and run " +
      "“deepvis: Visualize model here…”, or add dv.visualize(model, inputs).");
  }
  const m = models.length === 1 ? models[0] : await vscode.window.showQuickPick(
    models.map((x) => ({ label: x.name, description: `line ${x.line}`, ...x })), { placeHolder: "Which model?" });
  return m && visualizeModel(uri, m.name, m.line);
}

async function visualizeModel(uri, name, line, inputs) {
  const ed = vscode.window.activeTextEditor;
  uri = uri instanceof vscode.Uri ? uri : ed && ed.document.uri;
  if (!uri) return;
  const doc = await vscode.workspace.openTextDocument(uri);
  if (!name && ed && ed.document.uri.toString() === uri.toString()) {
    const sel = ed.selection;
    name = sel.isEmpty ? doc.getText(doc.getWordRangeAtPosition(sel.active, /[A-Za-z_][\w.]*/)) : doc.getText(sel).trim();
    line = sel.end.line + 1;
  }
  if (!name) {
    name = await vscode.window.showInputBox({ prompt: "The model to visualize (a name or expression)", placeHolder: "model" });
    if (!name) return;
    line = line || (ed ? ed.selection.active.line + 1 : 1);
  }
  if (inputs === undefined) inputs = await askInputs(doc, name);
  if (inputs === undefined) return;
  return run({ kind: "model", uri, model: name, inputs, line: line || 1 });
}

async function askInputs(doc, model, candidates) {
  const nb = doc.uri.scheme === "vscode-notebook-cell" ? notebookOf(doc.uri) : null;
  const key = `inputs:${(nb ? nb.uri : doc.uri).toString()}:${model}`;
  const remembered = ctx.workspaceState.get(key);
  if (!candidates && nb) {
    const k = await kernelObjects(nb);
    candidates = k.objs ? k.objs.filter((o) => o.kind === "input").map((o) => ({ label: o.name, description: shapeText(o) }))
      : P.inputCandidates(nb.getCells().map((c) => c.document.getText()).join("\n")).map((c) => ({ label: c.name, description: "in the notebook" }));
  }
  const found = candidates || P.inputCandidates(doc.getText()).map((c) => ({ label: c.name, description: `line ${c.line}` }));
  const base = remembered && !found.some((f) => f.label === remembered)
    ? [{ label: remembered, description: "last used" }, ...found] : found;
  if (remembered) base.sort((a, b) => Number(b.label === remembered) - Number(a.label === remembered));
  return new Promise((resolve) => {
    const qp = vscode.window.createQuickPick();
    qp.title = `Inputs for ${model}`;
    qp.placeholder = "what you would call it with: a tensor (x) or a tuple ((x, t)) — type any expression";
    qp.items = base;
    qp.onDidChangeValue((v) => {
      qp.items = v.trim() && !base.some((b) => b.label === v.trim())
        ? [{ label: v.trim(), description: "use this expression", alwaysShow: true }, ...base] : base;
    });
    let done = false;
    qp.onDidAccept(() => {
      const v = (qp.selectedItems[0] && qp.selectedItems[0].label) || qp.value.trim();
      if (!v) return;
      done = true;
      ctx.workspaceState.update(key, v);
      resolve(v);
      qp.hide();
    });
    qp.onDidHide(() => { if (!done) resolve(undefined); qp.dispose(); });
    qp.show();
  });
}

function stop() {
  if (!job) return;
  job.stopped = true;
  kill(job.child);
}

async function savePage() {
  if (!lastResult || !lastResult.ok) return vscode.window.showInformationMessage("deepvis: no page to save yet.");
  const folder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
  const name = `${(lastResult.title || "deepvis").replace(/[^\w.-]+/g, "_")}.html`;
  const to = await vscode.window.showSaveDialog({
    defaultUri: folder ? vscode.Uri.joinPath(folder.uri, name) : vscode.Uri.file(path.join(os.homedir(), name)),
    filters: { "HTML page": ["html"] }, saveLabel: "Save page",
  });
  if (!to) return;
  await vscode.workspace.fs.copy(vscode.Uri.file(lastResult.page), to, { overwrite: true });
  const open = await vscode.window.showInformationMessage(`deepvis: saved ${path.basename(to.fsPath)}`, "Open in browser");
  if (open) vscode.env.openExternal(to);
}

// ------------------------------------------------------------------------------------------------ notebooks
// In a notebook the objects already live in its kernel (trained weights and all): evaluate the model / inputs / call
// there instead of re-running anything, through the Jupyter extension's kernel API.
const notebookOf = (cellUri) => vscode.workspace.notebookDocuments.find((nb) =>
  nb.getCells().some((c) => c.document.uri.toString() === cellUri.toString()));

async function kernelFor(nb) {
  const ext = vscode.extensions.getExtension("ms-toolsai.jupyter");
  if (!ext) return { error: "the Jupyter extension isn't installed" };
  try {
    const api = ext.isActive ? ext.exports : await ext.activate();
    const kernel = api && api.kernels && await api.kernels.getKernel(nb.uri);
    return kernel ? { kernel } : { error: "the notebook's kernel isn't running, or deepvis wasn't allowed to use it" };
  } catch (e) {
    return { error: String(e.message || e) };
  }
}

/** Run Python in the kernel; returns {stdout, error}. The code prints one JSON line after MARK. */
async function execKernel(kernel, code) {
  const src = new vscode.CancellationTokenSource();
  const dec = new TextDecoder();
  let stdout = "", error = null;
  try {
    for await (const out of kernel.executeCode(code, src.token)) {
      for (const item of out.items) {
        const t = dec.decode(item.data);
        if (item.mime === "application/vnd.code.notebook.error") {
          try { const e = JSON.parse(t); error = `${e.name}: ${e.message}`; } catch (e2) { error = t; }
        } else if (/stdout|text\/plain/.test(item.mime)) stdout += t;
      }
    }
  } catch (e) {
    error = String(e.message || e);
  }
  return { stdout, error };
}

const MARK = "\u0000deepvis:";
const py = (v) => JSON.stringify(v);  // a JSON string is a valid Python string literal

/** The bundled deepvis's capture PROTOCOL (0 when not vendored, e.g. running from a checkout). */
let bundledProtocol = null;
function bundledPROTOCOL() {
  if (bundledProtocol === null) {
    try {
      const src = fs.readFileSync(path.join(ctx.extensionPath, "python", "vendor", "deepvis", "capture.py"), "utf8");
      const m = /^PROTOCOL = (\d+)/m.exec(src);
      bundledProtocol = m ? Number(m[1]) : 0;
    } catch (e) { bundledProtocol = 0; }
  }
  return bundledProtocol;
}

function kernelCode(body) {
  const vendor = path.join(ctx.extensionPath, "python", "vendor");
  // The kernel's own deepvis is used if it speaks the bundled copy's protocol; an older one is swapped for the
  // bundled copy unless the notebook already imported it (then the fix is to upgrade it).
  return [
    "def __deepvis_run():",
    "    import json, os, sys",
    ...(announce ? [`    os.environ["DEEPVIS_VSCODE"] = ${py(announce.url)}`] : []),
    "    had = 'deepvis' in sys.modules",
    "    try:",
    "        from deepvis import capture",
    "    except ImportError:",
    "        capture = None",
    `    if capture is None or getattr(capture, 'PROTOCOL', 0) < ${bundledPROTOCOL()}:`,
    "        if capture is not None and had:",
    "            raise RuntimeError('this kernel\\'s deepvis is older than the VS Code extension: run '",
    "                               '`%pip install -U deepvis` in the notebook and restart the kernel')",
    "        for m in [m for m in sys.modules if m == 'deepvis' or m.startswith('deepvis.')]:",
    "            del sys.modules[m]",
    `        sys.path.insert(0, ${py(vendor)})`,
    "        from deepvis import capture",
    ...body.map((l) => "    " + l),
    "__deepvis_run()",
    "del __deepvis_run",
  ].join("\n");
}

function parseKernel(r) {
  const i = r.stdout.lastIndexOf(MARK);
  if (i < 0) return null;
  try { return JSON.parse(r.stdout.slice(i + MARK.length).split("\n")[0]); } catch (e) { return null; }
}

// offline: the page inlines the bundled KaTeX, since the panel loads nothing from the network
const kwOptions = (req) => `, **json.loads(${py(JSON.stringify({ offline: true, ...(req.options || {}) }))})`;

/** What to run in the kernel for a request: {body, label} (lines for kernelCode) or {error}. */
function kernelRequest(req, cellText, out, context = "") {
  if (req.kind === "call") {
    const calls = P.visualizeCalls(cellText, context);
    const c = calls[Math.min(req.index, calls.length - 1)];
    if (!c) return { error: ["no dv.visualize(...) call in this cell", "the call is gone."] };
    return { body: [`print(${py(MARK)} + json.dumps(capture.kernel_call(${py(out)}, ${py(c.source)}${kwOptions(req)})))`], label: `cell · line ${c.first}` };
  }
  return {
    body: [`print(${py(MARK)} + json.dumps(capture.visualize_to(${py(out)}, ${py(req.model)}, ${py(req.inputs || "")}` +
      `${req.params ? `, params=${py(req.params)}` : ""}${kwOptions(req)})))`],
    label: req.model,
  };
}

async function runInKernel(req, cellDoc) {
  const nb = notebookOf(cellDoc.uri);
  lastReq = req;
  showPanel();
  if (!nb) return finish(req, { ok: false, title: "couldn't find this cell's notebook", message: "reopen the notebook and try again." });
  const k = await kernelFor(nb);
  if (k.error) {
    return finish(req, { ok: false, title: "deepvis needs the notebook's kernel",
      message: `${k.error}.\n  → fix: run a cell first (so the kernel starts) and allow deepvis to use it when asked — or add dv.visualize(model, x) to a cell to see the page inline.` });
  }
  const out = path.join(tmpDir, `page-${++pageCount}.html`);
  const spec = kernelRequest(req, cellDoc.getText(), out, contextOf(cellDoc));
  if (spec.error) return finish(req, { ok: false, title: spec.error[0], message: spec.error[1] });
  const { body, label } = spec;
  post({ type: "running", label: `running in the kernel · ${label}` });
  status.text = "$(sync~spin) deepvis: kernel";
  status.show();
  output.appendLine(`\n[deepvis] in the kernel of ${path.basename(nb.uri.fsPath)}: ${label}`);
  const r = await execKernel(k.kernel, kernelCode(body));
  let result = parseKernel(r);
  if (!result) {
    result = { ok: false, title: "the kernel didn't answer",
      message: `${r.error || r.stdout.slice(-600) || "no output"}\n  → fix: check that the kernel's Python can import deepvis and numpy.` };
  }
  if (r.stdout) output.append(r.stdout.replace(/\u0000deepvis:.*\n?/g, ""));
  return finish(req, result, cellDoc);
}

async function kernelObjects(nb) {
  const k = await kernelFor(nb);
  if (k.error) return { error: k.error };
  const r = await execKernel(k.kernel, kernelCode([`print(${py(MARK)} + json.dumps(capture.kernel_objects()))`]));
  const objs = parseKernel(r);
  return objs ? { objs } : { error: r.error || "the kernel didn't answer" };
}

const shapeText = (o) => o.kind === "input"
  ? (o.type === "tuple" ? `(${o.shape.map((s) => `(${s.join(", ")})`).join(", ")})` : `${o.type}(${o.shape.join(", ")})`)
  : `${o.type}${o.params ? ` · ${o.params.toLocaleString()} params` : ""}`;

async function visualizeLive() {
  const nbEd = vscode.window.activeNotebookEditor;
  if (!nbEd) return vscode.window.showInformationMessage("deepvis: open a notebook (with a running kernel) first.");
  const nb = nbEd.notebook;
  const found = await vscode.window.withProgress({ location: vscode.ProgressLocation.Window, title: "deepvis: asking the kernel" },
    () => kernelObjects(nb));
  if (found.error) return vscode.window.showWarningMessage(`deepvis: ${found.error}. Run a cell first, then try again.`);
  const models = found.objs.filter((o) => o.kind === "model");
  if (!models.length) return vscode.window.showInformationMessage("deepvis: no models in the kernel yet — run the cell that creates one.");
  const m = models.length === 1 ? models[0] : await vscode.window.showQuickPick(
    models.map((o) => ({ label: o.name, description: shapeText(o), o })), { placeHolder: "Which model?" }).then((p) => p && /** @type {any} */ (p).o);
  if (!m) return;
  const cell = nb.cellAt(nbEd.selection ? nbEd.selection.start : 0);
  const inputs = await askInputs(cell.document, m.name, found.objs.filter((o) => o.kind === "input")
    .map((o) => ({ label: o.name, description: shapeText(o) })));
  if (inputs === undefined) return;
  return run({ kind: "model", uri: cell.document.uri, model: m.name, inputs, line: 1 });
}

// ------------------------------------------------------------------------------------------------ running
async function pythonFor(uri) {
  const set = config(uri).get("pythonPath");
  if (set) return set;
  const ext = vscode.extensions.getExtension("ms-python.python");
  if (ext) {
    try {
      const api = ext.isActive ? ext.exports : await ext.activate();
      const envs = api && api.environments;
      if (envs && envs.getActiveEnvironmentPath) {
        const p = envs.getActiveEnvironmentPath(uri);
        const r = envs.resolveEnvironment ? await envs.resolveEnvironment(p) : null;
        const exe = r && r.executable && r.executable.uri && r.executable.uri.fsPath;
        if (exe) return exe;
        if (p && p.path && fs.existsSync(p.path) && fs.statSync(p.path).isFile()) return p.path;
      }
      const d = api && api.settings && api.settings.getExecutionDetails && api.settings.getExecutionDetails(uri);
      if (d && d.execCommand && d.execCommand[0]) return d.execCommand[0];
    } catch (e) {
      output.appendLine(`[deepvis] couldn't ask the Python extension for the interpreter: ${e}`);
    }
  }
  return process.platform === "win32" ? "python" : "python3";
}

/** The capture arguments for a request against the current text (calls and lines are re-found after edits). */
function captureArgs(req, text) {
  if (req.kind === "call") {
    const calls = P.visualizeCalls(text);
    const c = calls[Math.min(req.index, calls.length - 1)];
    if (!c) return { error: ["no dv.visualize(...) call in this file", "the call this panel showed is gone.\n  → fix: add dv.visualize(model, inputs), or use “Visualize model here…”."] };
    return { args: ["--lines", `${c.first}-${c.last}`, ...optionArgs(req)], label: `line ${c.first}` };
  }
  const line = P.evalLine(text, req.line, req.model, req.inputs);
  const args = ["--line", String(line), "--model", req.model, "--inputs", req.inputs || "", ...optionArgs(req)];
  if (req.params) args.push("--params", req.params);
  return { args, label: req.model };
}

const optionArgs = (req) => (req.options && Object.keys(req.options).length ? ["--options", JSON.stringify(req.options)] : []);

async function compareCheckpoint() {
  if (!lastReq) return vscode.window.showInformationMessage("deepvis: visualize a model first, then compare it with a checkpoint.");
  const pick = await vscode.window.showOpenDialog({
    canSelectMany: false, openLabel: "Compare with this",
    filters: { "Checkpoints": ["pt", "pth", "ckpt", "bin", "safetensors"], "All files": ["*"] },
  });
  if (!pick || !pick.length) return;
  const file = pick[0].fsPath;
  return run({ ...lastReq, options: { ...(lastReq.options || {}), compare: file, start: "compare",
    compare_names: ["current", path.basename(file)] } });
}

function kill(child) {
  if (!child || child.exitCode !== null) return;
  try {
    if (process.platform === "win32") child.kill(); else process.kill(-child.pid, "SIGTERM");
  } catch (e) { try { child.kill(); } catch (e2) { /* gone */ } }
  setTimeout(() => {
    if (child.exitCode === null) { try { process.platform === "win32" ? child.kill("SIGKILL") : process.kill(-child.pid, "SIGKILL"); } catch (e) { /* gone */ } }
  }, 2000);
}

async function run(req) {
  if (!req || !req.uri) return;
  const doc = await vscode.workspace.openTextDocument(req.uri);
  if (doc.uri.scheme === "vscode-notebook-cell") return runInKernel(req, doc);
  const text = doc.getText();
  lastReq = req;
  showPanel();
  const spec = captureArgs(req, text);
  if (spec.error) return finish(req, { ok: false, title: spec.error[0], message: spec.error[1] });
  if (job) { job.replaced = true; kill(job.child); }

  const python = await pythonFor(doc.uri);
  const script = doc.isUntitled ? path.join(tmpDir, "untitled.py") : doc.uri.fsPath;
  const folder = vscode.workspace.getWorkspaceFolder(doc.uri);
  const cwd = !config(doc.uri).get("runInFileDir") && folder ? folder.uri.fsPath : path.dirname(script);
  const out = path.join(tmpDir, `page-${++pageCount}.html`);
  const args = [path.join(ctx.extensionPath, "python", "run.py"), out, script, "--source", "-", ...spec.args];
  const name = path.basename(script);
  post({ type: "running", label: `running ${name} · ${spec.label}` });
  status.text = `$(sync~spin) deepvis: ${name}`;
  status.tooltip = "deepvis is running this file — click to stop";
  status.show();
  vscode.commands.executeCommand("setContext", "deepvis.running", true);
  output.appendLine(`\n[deepvis] ${python} ${args.slice(1).map((a) => (/\s/.test(a) ? JSON.stringify(a) : a)).join(" ")}`);

  return new Promise((resolve) => {
    let child;
    try {
      child = cp.spawn(python, args, {
        cwd: fs.existsSync(cwd) ? cwd : os.homedir(), detached: process.platform !== "win32",
        env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8", MPLBACKEND: "Agg", DEEPVIS_OFFLINE: "1",
          ...(announce ? { DEEPVIS_VSCODE: announce.url } : {}) },
      });
    } catch (e) {
      return resolve(finish(req, noPython(python, e)));
    }
    const me = { child, req, stopped: false, replaced: false, stderr: "", resolve };
    job = me;
    const seconds = config(doc.uri).get("timeoutSeconds") || 300;
    const timer = me.timer = setTimeout(() => { me.timedOut = true; kill(child); }, seconds * 1000);
    child.stdout.on("data", (d) => output.append(d.toString()));
    child.stderr.on("data", (d) => { const s = d.toString(); me.stderr += s; output.append(s); });
    child.stdin.on("error", () => { /* the process ended before reading its input */ });
    child.stdin.end(text);
    let spawnError = null;
    child.on("error", (e) => { spawnError = e; });
    child.on("close", (code, signal) => {
      clearTimeout(timer);
      if (job === me) job = null;
      if (me.replaced) return resolve(lastResult);  // a newer run took over
      if (me.live) {  // the live run ended (stopped, or its script finished)
        status.hide();
        vscode.commands.executeCommand("setContext", "deepvis.running", false);
        if (live && live.pid === child.pid) { live = null; post({ type: "notice", text: "the live run ended — this is its last state" }); }
        return;
      }
      let result;
      if (spawnError) result = noPython(python, spawnError);
      else if (me.stopped) result = { ok: false, title: "stopped", message: "the run was stopped." };
      else if (me.timedOut) {
        result = { ok: false, title: `the run took longer than ${seconds} s and was stopped`,
          message: "  → fix: use a smaller model or input, stop long training before the model is visualized, or raise the deepvis.timeoutSeconds setting.", setting: "deepvis.timeoutSeconds" };
      } else {
        result = readResult(out, code, signal, me.stderr);
      }
      resolve(finish(req, result, doc));
    });
  });
}

function noPython(python, e) {
  return { ok: false, title: "couldn't start Python",
    message: `${python}: ${e.code === "ENOENT" ? "not found" : e.message}\n  → fix: pick an interpreter (Python: Select Interpreter) or set the deepvis.pythonPath setting.`,
    action: { label: "Select interpreter", command: "python.setInterpreter" } };
}

function readResult(out, code, signal, stderr) {
  try {
    const r = JSON.parse(fs.readFileSync(out + ".json", "utf8"));
    if (r.ok && r.live) return r;
    if (r.ok && !fs.existsSync(r.page)) return { ok: false, title: "no page was written", message: `expected ${r.page}` };
    return r;
  } catch (e) {
    if (signal || code === null || code < 0) {
      return { ok: false, title: "Python was stopped by the system",
        message: `it ended with ${signal || code} — most likely out of memory.\n  → fix: use a smaller model or input.` };
    }
    const tail = stderr.split("\n").filter((l) => l.trim() && !/Warning/.test(l)).slice(-12).join("\n");
    return { ok: false, title: "the run failed", message: tail || `Python exited with code ${code}.` };
  }
}

function finish(req, result, doc) {
  status.hide();
  vscode.commands.executeCommand("setContext", "deepvis.running", false);
  const uri = req.uri;
  diagnostics.delete(uri);
  if (result.ok && result.live) {
    lastResult = result;
    return result;
  }
  if (result.ok) {
    live = null;
    let html = fs.readFileSync(result.page, "utf8");
    const m = /<title>([^<]*?)(?: · deepvis)?<\/title>/.exec(html);
    result.title = m ? decode(m[1]) : "deepvis";
    html = injectBridge(html);
    if (panel) {
      panel.title = `deepvis · ${result.title}`;
      post({ type: "page", html });
    }
  } else {
    if (result.line && doc && result.line <= doc.lineCount) {
      const ln = doc.lineAt(result.line - 1);
      const d = new vscode.Diagnostic(new vscode.Range(result.line - 1, ln.firstNonWhitespaceCharacterIndex, result.line - 1, ln.text.length),
        `${result.title}: ${result.message}`, vscode.DiagnosticSeverity.Error);
      d.source = "deepvis";
      diagnostics.set(uri, [d]);
    }
    post({ type: "error", title: result.title, message: result.message, line: result.line || null,
      action: result.action ? result.action.label : null });
  }
  lastResult = result;
  return result;
}

const decode = (s) => s.replace(/&amp;/g, "&").replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, '"').replace(/&#x27;|&#39;/g, "'");

// ------------------------------------------------------------------------------------------------ the panel
function showPanel() {
  if (panel) { if (!panel.visible) panel.reveal(vscode.ViewColumn.Beside, true); return; }
  panel = vscode.window.createWebviewPanel(VIEW, "deepvis", { viewColumn: vscode.ViewColumn.Beside, preserveFocus: true },
    { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [] });
  panel.iconPath = vscode.Uri.file(path.join(ctx.extensionPath, "media", "icon.png"));
  panel.webview.html = shell();
  panel.webview.onDidReceiveMessage(onMessage);
  panel.onDidDispose(() => { panel = null; stop(); });
}

const post = (msg) => panel && panel.webview.postMessage(msg);

async function onMessage(m) {
  if (m.type === "stop") stop();
  else if (m.type === "rerun") vscode.commands.executeCommand("deepvis.rerun");
  else if (m.type === "output") output.show(true);
  else if (m.type === "action" && lastResult && lastResult.action) vscode.commands.executeCommand(lastResult.action.command);
  else if (m.type === "goto" && lastReq) {
    const doc = await vscode.workspace.openTextDocument(lastReq.uri);
    const line = Math.max(0, Math.min((m.line || 1) - 1, doc.lineCount - 1));
    const ed = await vscode.window.showTextDocument(doc, { viewColumn: vscode.ViewColumn.One });
    ed.selection = new vscode.Selection(line, doc.lineAt(line).firstNonWhitespaceCharacterIndex, line, doc.lineAt(line).text.length);
    ed.revealRange(ed.selection, vscode.TextEditorRevealType.InCenterIfOutsideViewport);
  } else if (m.type === "open" && /^https?:/.test(m.url || "")) vscode.env.openExternal(vscode.Uri.parse(m.url));
  else if (m.type === "download") {
    const folder = vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders[0];
    const name = String(m.name || "download").replace(/[\\/]/g, "_");
    const to = await vscode.window.showSaveDialog({
      defaultUri: folder ? vscode.Uri.joinPath(folder.uri, name) : vscode.Uri.file(path.join(os.homedir(), name)), saveLabel: "Save",
    });
    if (!to) return;
    const data = String(m.data || "");
    const b64 = data.slice(data.indexOf(",") + 1);
    await vscode.workspace.fs.writeFile(to, Buffer.from(b64, "base64"));
    vscode.window.showInformationMessage(`deepvis: saved ${path.basename(to.fsPath)}`);
  }
}

// Runs inside the page: webviews can't download files or open links, so hand those to the extension.
const BRIDGE = `<script>(()=>{const up=(m)=>parent.postMessage(Object.assign({__deepvis:1},m),"*");
const click=HTMLAnchorElement.prototype.click;
HTMLAnchorElement.prototype.click=function(){if(!this.hasAttribute("download"))return click.call(this);
const name=this.getAttribute("download")||"download";fetch(this.href).then((r)=>r.blob()).then((b)=>{const f=new FileReader();
f.onload=()=>up({type:"download",name,data:f.result});f.readAsDataURL(b);});};
document.addEventListener("click",(e)=>{const a=e.target&&e.target.closest&&e.target.closest("a[href]");if(!a)return;
if(a.hasAttribute("download")){e.preventDefault();a.click();return;}
const h=a.getAttribute("href")||"";if(/^https?:/.test(h)){e.preventDefault();up({type:"open",url:h});}},true);})();</script>`;

function injectBridge(html) {
  const i = html.search(/<head[^>]*>/i);
  if (i < 0) return BRIDGE + html;
  const j = html.indexOf(">", i) + 1;
  return html.slice(0, j) + BRIDGE + html.slice(j);
}

function shell() {
  const csp = [
    "default-src 'none'",
    "script-src 'unsafe-inline'",  // pages inline everything they run (KaTeX included: DEEPVIS_OFFLINE)
    "style-src 'unsafe-inline'",
    "font-src data:",
    "img-src data: blob: https:",
    "connect-src blob: data:",
    "worker-src blob:",
    "frame-src 'self' about: data: blob: http://127.0.0.1:* http://localhost:* https:",
  ].join("; ");
  return `<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="${csp}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; background: #000; color: #DDD; overflow: hidden;
    font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif; }
  #page { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; background: #000; }
  #bar { position: absolute; left: 0; right: 0; top: 0; height: 2px; z-index: 5; overflow: hidden; display: none; }
  #bar::before { content: ""; position: absolute; top: 0; bottom: 0; width: 30%; background: #58C4DD;
    animation: slide 1.1s ease-in-out infinite; }
  @keyframes slide { from { left: -30%; } to { left: 100%; } }
  .running #bar { display: block; }
  #chip { position: absolute; right: 12px; bottom: 12px; z-index: 5; display: none; align-items: center; gap: 10px;
    background: rgba(0,0,0,.88); border: 1px solid #444; border-radius: 999px; padding: 5px 6px 5px 12px; font-size: 12px; color: #BBB; }
  .running #chip { display: flex; }
  button { font: inherit; color: #DDD; background: #111; border: 1px solid #444; border-radius: 8px; padding: 4px 10px; cursor: pointer; }
  button:hover { border-color: #888; background: #222; }
  button:focus-visible { outline: 2px solid #FFFF00; outline-offset: 2px; }
  #chip button { border-radius: 999px; padding: 2px 10px; }
  #notice { position: absolute; left: 50%; top: 10px; transform: translateX(-50%); z-index: 6; background: rgba(0,0,0,.9);
    border: 1px solid #444; border-radius: 999px; padding: 5px 14px; font-size: 12px; color: #BBB; }
  #notice[hidden] { display: none; }
  #empty { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; flex-direction: column;
    gap: 14px; color: #888; }
  #empty svg { opacity: .9; }
  .has-page #empty { display: none; }
  body:not(.has-page) #page { visibility: hidden; }
  #err { position: absolute; left: 16px; right: 16px; bottom: 16px; max-width: 760px; margin: 0 auto; z-index: 6; display: none;
    background: #120606; border: 1px solid #FC6255; border-radius: 10px; padding: 14px 16px; box-shadow: 0 8px 30px rgba(0,0,0,.6); }
  .error #err { display: block; }
  .error:not(.has-page) #err { top: 50%; bottom: auto; transform: translateY(-50%); }
  #err .t { color: #FC6255; font-weight: 600; margin-bottom: 8px; padding-right: 28px; }
  #err pre { white-space: pre-wrap; font: 12px/1.5 ui-monospace, "SF Mono", Menlo, Consolas, monospace; color: #DDD; margin: 0 0 12px; max-height: 40vh; overflow: auto; }
  #err .acts { display: flex; gap: 8px; flex-wrap: wrap; }
  #err .x { position: absolute; top: 8px; right: 8px; border: 0; background: none; color: #888; font-size: 16px; padding: 2px 8px; }
  #err button.primary { border-color: #1C758A; color: #C7E9F1; background: rgba(28,117,138,.28); }
</style></head>
<body>
  <div id="bar"></div>
  <div id="empty">
    <svg width="44" height="44" viewBox="0 0 24 24"><g stroke="#888" stroke-width="1.2"><line x1="5" y1="6" x2="18" y2="12"/><line x1="5" y1="18" x2="18" y2="12"/></g>
      <circle cx="5" cy="6" r="2.6" fill="#DDD"/><circle cx="5" cy="18" r="2.6" fill="#888"/><circle cx="18" cy="12" r="3.2" fill="#FFFF00"/></svg>
    <div>running your code…</div>
  </div>
  <iframe id="page" title="deepvis page"></iframe>
  <div id="notice" hidden><span id="noticeText"></span></div>
  <div id="chip"><span id="chipText">running…</span><button id="stop" title="Stop the run">Stop</button></div>
  <div id="err" role="alert"><button class="x" id="errClose" title="Dismiss">×</button><div class="t" id="errTitle"></div><pre id="errMsg"></pre>
    <div class="acts"><button class="primary" id="goto">Go to line</button><button id="action"></button><button id="rerun">Run again</button><button id="output">Show output</button></div></div>
<script>
  const vscode = acquireVsCodeApi();
  const $ = (id) => document.getElementById(id), body = document.body, frame = $("page");
  $("stop").onclick = () => vscode.postMessage({ type: "stop" });
  $("rerun").onclick = () => vscode.postMessage({ type: "rerun" });
  $("output").onclick = () => vscode.postMessage({ type: "output" });
  $("action").onclick = () => vscode.postMessage({ type: "action" });
  $("errClose").onclick = () => body.classList.remove("error");
  let gotoLine = null;
  $("goto").onclick = () => vscode.postMessage({ type: "goto", line: gotoLine });
  window.addEventListener("message", (e) => {
    const m = e.data || {};
    if (m.__deepvis) { vscode.postMessage(m); return; }          // from the page: downloads, links
    if (m.type === "running") { body.classList.add("running"); $("chipText").textContent = m.label; }
    else if (m.type === "page") {
      body.classList.remove("running", "error", "is-live"); body.classList.add("has-page");
      frame.removeAttribute("src");
      frame.srcdoc = m.html;
    } else if (m.type === "live") {
      body.classList.remove("running", "error"); body.classList.add("has-page", "is-live");
      frame.removeAttribute("srcdoc");
      frame.src = m.url;
    } else if (m.type === "notice") {
      body.classList.remove("running");
      $("noticeText").textContent = m.text; $("notice").hidden = false;
      clearTimeout(window.__nt); window.__nt = setTimeout(() => { $("notice").hidden = true; }, 6000);
    } else if (m.type === "error") {
      body.classList.remove("running"); body.classList.add("error");
      $("errTitle").textContent = m.title; $("errMsg").textContent = m.message || "";
      gotoLine = m.line; $("goto").style.display = m.line ? "" : "none"; $("goto").textContent = "Go to line " + m.line;
      $("action").style.display = m.action ? "" : "none"; $("action").textContent = m.action || "";
      if (!body.classList.contains("has-page")) $("empty").querySelector("div").textContent = "no page yet";
    }
  });
</script></body></html>`;
}

module.exports = { activate, deactivate, _internals: { shell, injectBridge, kernelCode, kernelRequest, parseKernel, MARK, setContext: (c) => { ctx = c; } } };  // _internals: for tests
