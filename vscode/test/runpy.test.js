// python/run.py picks the user's installed deepvis only if it speaks the bundled copy's capture PROTOCOL.
//   DEEPVIS_TEST_PYTHON=/path/to/python node test/runpy.test.js      (after `npm run vendor`)
"use strict";
const assert = require("assert");
const cp = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const root = path.resolve(__dirname, "..");
const python = process.env.DEEPVIS_TEST_PYTHON || "python3";
const vendored = path.join(root, "python", "vendor", "deepvis", "capture.py");
if (!fs.existsSync(vendored)) { console.log("run.py: skipped (run `npm run vendor` first)"); process.exit(0); }
const bundled = Number(/^PROTOCOL = (\d+)/m.exec(fs.readFileSync(vendored, "utf8"))[1]);

/** A fake installed deepvis whose capture.main writes `fake` to OUT.html.json. */
function fakeInstall(protocol) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "deepvis-runpy-"));
  fs.mkdirSync(path.join(dir, "deepvis"));
  fs.writeFileSync(path.join(dir, "deepvis", "__init__.py"), '__version__ = "0.0.1"\n');
  fs.writeFileSync(path.join(dir, "deepvis", "capture.py"), [
    ...(protocol === null ? [] : [`PROTOCOL = ${protocol}`]),
    "import json",
    "def main(argv):",
    "    open(argv[0] + '.json', 'w').write(json.dumps({'ok': False, 'title': 'fake'}))",
  ].join("\n") + "\n");
  fs.writeFileSync(path.join(dir, "script.py"), "x = 1\n");
  return dir;
}

function runWith(dir) {
  const out = path.join(dir, "page.html");
  const r = cp.spawnSync(python, [path.join(root, "python", "run.py"), out, path.join(dir, "script.py")],
    { env: { ...process.env, PYTHONPATH: dir }, encoding: "utf8" });
  return { stdout: r.stdout, result: JSON.parse(fs.readFileSync(out + ".json", "utf8")) };
}

let failed = 0;
function check(name, fn) {
  try { fn(); console.log(`  ✓ ${name}`); } catch (e) { failed++; console.log(`  ✗ ${name}\n    ${e.stack || e}`); }
}

console.log("run.py");
check("an installed deepvis without PROTOCOL falls back to the bundled copy", () => {
  const { stdout, result } = runWith(fakeInstall(null));
  assert.notStrictEqual(result.title, "fake", "the old install ran");
  assert.match(stdout, /older than this extension .*using the bundled copy/);
});
check("an older PROTOCOL falls back too", () => {
  const { result } = runWith(fakeInstall(bundled - 1));
  assert.notStrictEqual(result.title, "fake");
});
check("an installed deepvis with the same or a newer PROTOCOL is used", () => {
  for (const p of [bundled, bundled + 1]) {
    const { stdout, result } = runWith(fakeInstall(p));
    assert.strictEqual(result.title, "fake");
    assert.match(stdout, /using deepvis 0\.0\.1/);
  }
});
process.exit(failed ? 1 : 0);
