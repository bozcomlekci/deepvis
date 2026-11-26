// Run the integration suite in a real VS Code: a fresh profile, the fixtures as the workspace, this folder as the
// extension under development. DEEPVIS_TEST_PYTHON picks the interpreter (default: python3); VSCODE_BIN the app.
"use strict";
const cp = require("child_process");
const fs = require("fs");
const os = require("os");
const path = require("path");

const root = path.resolve(__dirname, "..");

/** VS Code to test in: $VSCODE_BIN, the installed app on macOS, else a download (CI). */
async function vscodeBin() {
  if (process.env.VSCODE_BIN) return process.env.VSCODE_BIN;
  const mac = "/Applications/Visual Studio Code.app/Contents/MacOS/Code";
  if (process.platform === "darwin" && fs.existsSync(mac)) return mac;
  const { downloadAndUnzipVSCode } = require("@vscode/test-electron");
  const bin = await downloadAndUnzipVSCode("stable");
  if (fs.existsSync(bin)) return bin;
  // VS Code 1.110+ on macOS names its binary after the product (Contents/MacOS/Code), not "Electron"
  const dir = path.dirname(bin);
  const named = fs.existsSync(dir) ? fs.readdirSync(dir).filter((f) => fs.statSync(path.join(dir, f)).isFile()) : [];
  if (named.length === 1) return path.join(dir, named[0]);
  throw new Error(`no VS Code binary at ${bin} (found: ${named.join(", ") || "nothing"})`);
}

async function main() {
const bin = await vscodeBin();
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "deepvis-vscode-test-"));
const ws = path.join(tmp, "ws");
fs.cpSync(path.join(__dirname, "fixtures"), ws, { recursive: true });
fs.mkdirSync(path.join(ws, ".vscode"));
fs.writeFileSync(path.join(ws, ".vscode", "settings.json"), JSON.stringify({
  "deepvis.pythonPath": process.env.DEEPVIS_TEST_PYTHON || "python3", "deepvis.refreshOnSave": false,
  "workbench.startupEditor": "none", "security.workspace.trust.enabled": false,
}, null, 2));
const shot = process.env.DEEPVIS_TEST_SHOT;  // a .png path: hold the window open and screenshot it
const env = { ...process.env, DEEPVIS_TEST_WS: ws, DEEPVIS_TEST_HOLD: shot ? "1" : "" };
delete env.ELECTRON_RUN_AS_NODE;
const child = cp.spawn(bin, [
  `--extensionDevelopmentPath=${root}`, `--extensionTestsPath=${path.join(__dirname, "suite")}`,
  `--user-data-dir=${path.join(tmp, "user")}`, `--extensions-dir=${path.join(tmp, "ext")}`,
  "--disable-extensions", "--skip-welcome", "--skip-release-notes", "--new-window", ws,
  ...(process.platform === "linux" ? ["--no-sandbox"] : []),  // CI runners lack a configured SUID sandbox
], { env, stdio: "inherit" });
if (shot) {
  const ready = path.join(ws, ".ready");
  const poll = setInterval(() => {
    if (!fs.existsSync(ready)) return;
    clearInterval(poll);
    setTimeout(() => {
      cp.execFileSync("screencapture", ["-x", "-o", shot]);
      fs.writeFileSync(path.join(ws, ".shot-done"), "");
      console.log(`screenshot: ${shot}`);
    }, 1500);
  }, 300);
}
child.on("exit", (code) => { fs.rmSync(tmp, { recursive: true, force: true }); process.exit(code === null ? 1 : code); });
}

main().catch((e) => { console.error(e); process.exit(1); });
