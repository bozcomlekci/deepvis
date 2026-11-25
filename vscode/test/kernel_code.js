// Print the Python the extension sends to a notebook kernel, for tests/test_live_editor.py:
//   node test/kernel_code.js '<request json>' OUT.html '<cell text>'
"use strict";
const Module = require("module");
const orig = Module._load;
Module._load = function (r, ...a) { return r === "vscode" ? { EventEmitter: class { constructor() { this.event = () => {}; } } } : orig.call(this, r, ...a); };
const path = require("path");
const { _internals: I } = require("../extension.js");
I.setContext({ extensionPath: path.resolve(__dirname, "..") });
const [req, out, cell] = process.argv.slice(2);
const spec = req === "objects" ? { body: [`print(${JSON.stringify(I.MARK)} + json.dumps(capture.kernel_objects()))`] }
  : I.kernelRequest(JSON.parse(req), cell || "", out);
process.stdout.write(JSON.stringify({ code: I.kernelCode(spec.body), mark: I.MARK }));
