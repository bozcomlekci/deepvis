// Integration tests, run inside VS Code by test/run.js.
"use strict";
const assert = require("assert");
const fs = require("fs");
const path = require("path");
const vscode = require("vscode");

const http = require("http");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const fetchJSON = (url) => new Promise((resolve, reject) => http.get(url, (res) => {
  let b = ""; res.on("data", (d) => { b += d; }); res.on("end", () => resolve(JSON.parse(b)));
}).on("error", reject));
const postJSON = (url, body) => new Promise((resolve, reject) => {
  const req = http.request(url, { method: "POST", headers: { "Content-Type": "application/json" } }, (res) => { res.resume(); resolve(res.statusCode); });
  req.on("error", reject); req.end(JSON.stringify(body));
});

async function check(name, fn) {
  const t0 = Date.now();
  try { await fn(); console.log(`  ✓ ${name} (${Date.now() - t0} ms)`); } catch (e) { console.log(`  ✗ ${name}\n    ${e.stack || e}`); throw e; }
}

exports.run = async function run() {
  const ws = process.env.DEEPVIS_TEST_WS;
  const api = await vscode.extensions.getExtension("batuoz.deepvis").activate();
  const model = vscode.Uri.file(path.join(ws, "model.py"));
  const bad = vscode.Uri.file(path.join(ws, "bad.py"));
  await vscode.window.showTextDocument(model);
  console.log("deepvis extension");

  await check("code lenses on calls and on the model line", async () => {
    const lenses = await vscode.commands.executeCommand("vscode.executeCodeLensProvider", model);
    const titles = lenses.map((l) => `${l.range.start.line + 1}:${l.command.title}`);
    assert.deepStrictEqual(titles.sort(), ["16:$(graph) Visualize net…", "18:$(graph) Visualize", "19:$(graph) Visualize"].sort());
  });
  await check("a chosen dv.visualize call (the second one, over two lines)", async () => {
    const r = await vscode.commands.executeCommand("deepvis.visualizeCall", model, 1);
    assert.ok(r.ok, r.message); assert.strictEqual(r.title, "second");
    assert.ok(api.panel(), "the panel is open");
  });
  await check("a model and inputs picked in the code (no call needed)", async () => {
    const r = await vscode.commands.executeCommand("deepvis.visualizeModel", model, "net", 16, "x");
    assert.ok(r.ok, r.message); assert.strictEqual(r.title, "Net");
  });
  await check("unsaved edits are what runs", async () => {
    const ed = await vscode.window.showTextDocument(model);
    await ed.edit((e) => e.replace(new vscode.Range(17, 0, 17, 999), 'dv.visualize(net, x, title="edited")'));
    const r = await api.run({ kind: "call", uri: model, index: 0 });
    assert.ok(r.ok, r.message); assert.strictEqual(r.title, "edited");
  });
  await check("a failing script: explained, with a squiggle on the line", async () => {
    await vscode.window.showTextDocument(bad);
    const r = await api.run({ kind: "call", uri: bad, index: 0 });
    assert.ok(!r.ok); assert.strictEqual(r.line, 7, JSON.stringify(r));
    assert.match(r.message, /expects 4 features/);
    const d = vscode.languages.getDiagnostics(bad);
    assert.strictEqual(d.length, 1); assert.strictEqual(d[0].range.start.line, 6);
  });
  await check("fixed → the squiggle goes away", async () => {
    const ed = await vscode.window.showTextDocument(bad);
    await ed.edit((e) => e.replace(new vscode.Range(5, 0, 5, 999), "x = torch.randn(3, 4)"));
    const r = await api.run({ kind: "call", uri: bad, index: 0 });
    assert.ok(r.ok, r.message);
    assert.strictEqual(vscode.languages.getDiagnostics(bad).length, 0);
  });
  await check("a name that doesn't exist yet", async () => {
    const r = await api.run({ kind: "model", uri: model, model: "nett", inputs: "x", line: 16 });
    assert.ok(!r.ok); assert.match(r.message, /nett/);
  });

  await check("a live run (dv.visualize(..., live=True)) opens in the panel and keeps serving", async () => {
    await api.ready;
    const live = vscode.Uri.file(path.join(ws, "live.py"));
    const r = await api.run({ kind: "call", uri: live, index: 0 });
    assert.ok(r.ok && r.live, JSON.stringify(r));
    assert.strictEqual(api.live().title, "live fixture");
    const st = await fetchJSON(r.live + "api/version");
    assert.ok(st.step >= 0);
    await vscode.commands.executeCommand("deepvis.stop");
    for (let i = 0; i < 50 && api.live(); i++) await sleep(100);
    assert.strictEqual(api.live(), null, "the panel noticed the run ended");
  });
  await check("a live page started in a terminal announces itself", async () => {
    const res = await postJSON(api.announceUrl() + "/announce", { url: "http://127.0.0.1:65000/", title: "from a terminal", pid: 1 });
    assert.strictEqual(res, 200);
    for (let i = 0; i < 20 && !api.live(); i++) await sleep(50);
    assert.strictEqual(api.live().title, "from a terminal");
    assert.strictEqual(await postJSON(api.announceUrl() + "/announce", { url: "file:///etc/passwd" }), 400);
    assert.strictEqual(await postJSON(api.announceUrl().replace(/\/[^/]+$/, "/wrongtoken") + "/announce", { url: "http://127.0.0.1:1/" }), 404);
    await postJSON(api.announceUrl() + "/announce", { url: "http://127.0.0.1:65000/", closed: true });
    for (let i = 0; i < 20 && api.live(); i++) await sleep(50);
    assert.strictEqual(api.live(), null);
  });

  if (process.env.DEEPVIS_TEST_HOLD) {
    await vscode.commands.executeCommand("workbench.action.closeAllEditors");
    await vscode.window.showTextDocument(model, { viewColumn: vscode.ViewColumn.One });
    const r = await vscode.commands.executeCommand("deepvis.visualizeCall", model, 0);
    assert.ok(r.ok, r.message);
    await sleep(3000);
    fs.writeFileSync(path.join(ws, ".ready"), "");
    for (let i = 0; i < 100 && !fs.existsSync(path.join(ws, ".shot-done")); i++) await sleep(200);
  }
};
