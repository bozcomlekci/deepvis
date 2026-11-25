// Bundle the deepvis Python package with the extension (python/vendor/deepvis), for interpreters without it.
"use strict";
const fs = require("fs");
const path = require("path");

const src = path.resolve(__dirname, "..", "..", "src", "deepvis");
const dst = path.resolve(__dirname, "..", "python", "vendor", "deepvis");
if (!fs.existsSync(src)) {
  if (fs.existsSync(dst)) { console.log("vendor: using the existing bundled copy"); process.exit(0); }
  console.error(`vendor: ${src} not found`); process.exit(1);
}
fs.rmSync(dst, { recursive: true, force: true });
fs.cpSync(src, dst, { recursive: true, filter: (p) => !p.includes("__pycache__") && !p.endsWith(".pyc") });
console.log(`vendor: ${path.relative(process.cwd(), dst)}`);
