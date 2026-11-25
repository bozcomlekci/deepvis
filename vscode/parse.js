// Finding things in Python source without running it: dv.visualize calls, lines that build a model, likely inputs.
// Pure functions of the text (no vscode import), so they can be tested with plain node. Lines are 1-based.
// @ts-check
"use strict";

const IDENT = /^[A-Za-z_]\w*$/;

/** The names `visualize` is reachable under: "dv.visualize" for `import deepvis as dv`, "visualize" for
 * `from deepvis import visualize`, … */
function visualizeNames(text) {
  const names = new Set();
  for (const m of text.matchAll(/^[ \t]*import[ \t]+deepvis(?:[ \t]+as[ \t]+(\w+))?[ \t]*(?:#.*)?$/gm)) names.add(`${m[1] || "deepvis"}.visualize`);
  for (const m of text.matchAll(/^[ \t]*from[ \t]+deepvis[ \t]+import[ \t]+\(?([^)\n#]*)/gm)) {
    for (const part of m[1].split(",")) {
      const [name, alias] = part.trim().split(/\s+as\s+/);
      if (name === "visualize") names.add(alias || "visualize");
    }
  }
  return [...names];
}

/** Index of the paren closing the one at `open` (skipping strings and comments), or -1. */
function closeParen(text, open) {
  let depth = 0;
  for (let i = open; i < text.length; i++) {
    const c = text[i];
    if (c === "#") { while (i < text.length && text[i] !== "\n") i++; continue; }
    if (c === "'" || c === '"') {
      const triple = text.startsWith(c.repeat(3), i);
      const end = triple ? c.repeat(3) : c;
      i += end.length;
      while (i < text.length && !text.startsWith(end, i)) { if (text[i] === "\\") i++; if (!triple && text[i] === "\n") break; i++; }
      i += end.length - 1;
      continue;
    }
    if (c === "(" || c === "[" || c === "{") depth++;
    else if (c === ")" || c === "]" || c === "}") { depth--; if (depth === 0) return i; }
  }
  return -1;
}

const lineAt = (text, index) => text.slice(0, index).split("\n").length;
const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Every `dv.visualize(...)` call in `text`: [{first, last, text, source}] (lines it's written on, one-line text, raw
 * source). `context` is the text whose imports name deepvis (a whole notebook for one cell). */
function visualizeCalls(text, context = text) {
  // context: where the imports are (a notebook's cells together); with none found, the usual `dv` alias
  const found = visualizeNames(context);
  const names = found.length ? found : (context === text ? [] : ["dv.visualize"]);
  if (!names.length) return [];
  const re = new RegExp(`(^|[^\\w.])(${names.map(escape).join("|")})[ \\t]*\\(`, "gm");
  const calls = [];
  for (const m of text.matchAll(re)) {
    const start = m.index + m[1].length;
    const lineStart = text.lastIndexOf("\n", start) + 1;
    const before = text.slice(lineStart, start);
    if (before.includes("#") || /^\s*def\s/.test(before)) continue;  // a comment, or a definition
    const open = m.index + m[0].length - 1;
    const close = closeParen(text, open);
    const first = lineAt(text, start), last = close < 0 ? first : lineAt(text, close);
    const source = text.slice(start, close < 0 ? open + 1 : close + 1);
    calls.push({ first, last, text: source.replace(/\s+/g, " "), source });
  }
  return calls;
}

/** Classes defined in the file that look like networks (subclass something ending in Module / Model / Net). */
function moduleClasses(text) {
  const out = new Set();
  for (const m of text.matchAll(/^[ \t]*class[ \t]+(\w+)[ \t]*\(([^)]*)\)/gm)) {
    if (/(^|[.\s,])(Module|Model|Net|PreTrainedModel|nn\.Module|nnx\.Module|eqx\.Module)\b|Module\b/.test(m[2])) out.add(m[1]);
  }
  return out;
}

const MODEL_RHS = [
  /^(?:torch\.)?nn\.\w+\s*\(/, /^Sequential\s*\(/, /\.from_pretrained\s*\(/, /^(?:torchvision\.)?models\.\w+\s*\(/,
  /^timm\.create_model\s*\(/, /^nnx\.\w+\s*\(/, /^eqx\.nn\.\w+\s*\(/, /^hk\.transform/,
];
const NOT_MODEL = /^(?:nn\.functional|nn\.init|nn\.utils|nn\.Parameter)\b/;

/** Lines that create a model: [{name, line}]. */
function modelLines(text) {
  const classes = moduleClasses(text);
  const found = [];
  const lines = text.split("\n");
  lines.forEach((ln, i) => {
    const m = /^[ \t]*([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)[ \t]*(?::[^=]+)?=[ \t]*([^=].*)$/.exec(ln);
    if (!m || m[1].startsWith("self.")) return;  // a layer inside a model's __init__, not a model
    const rhs = m[2].trim();
    if (NOT_MODEL.test(rhs)) return;
    const head = /^([A-Za-z_]\w*)\s*\(/.exec(rhs);
    const isModel = MODEL_RHS.some((r) => r.test(rhs)) || (head && classes.has(head[1]));
    if (isModel) found.push({ name: m[1], line: i + 1 });
  });
  return found;
}

const TENSOR_RHS = /^(?:torch\.(?:randn|rand|randint|zeros|ones|full|tensor|as_tensor|from_numpy|arange|linspace|stack|cat|load)|jnp\.\w+|jax\.random\.\w+|np\.random\.\w+|np\.(?:zeros|ones|array|load)|tokenizer\s*\(|\w+\.(?:encode|tokenize))/;

/** Variables that look like model inputs (tensors), in order of appearance: [{name, line}]. */
function inputCandidates(text) {
  const seen = new Map();
  text.split("\n").forEach((ln, i) => {
    const m = /^[ \t]*([A-Za-z_]\w*)[ \t]*(?::[^=]+)?=[ \t]*([^=].*)$/.exec(ln);
    if (m && TENSOR_RHS.test(m[2].trim()) && !seen.has(m[1])) seen.set(m[1], i + 1);
    const t = /^[ \t]*([A-Za-z_]\w*)[ \t]*,[ \t]*([A-Za-z_]\w*)[ \t]*=[ \t]*(?:next\(iter|[A-Za-z_]\w*\[)/.exec(ln);  // x, y = next(iter(loader))
    if (t && !seen.has(t[1])) seen.set(t[1], i + 1);
  });
  return [...seen].map(([name, line]) => ({ name, line }));
}

/** Plain names used in a Python expression (`model.encoder`, `(x, t)` → model, x, t). */
function namesIn(expr) {
  const stripped = expr.replace(/(["'])(?:\\.|(?!\1).)*\1/g, "");
  const out = new Set();
  for (const m of stripped.matchAll(/(^|[^\w.])([A-Za-z_]\w*)/g)) {
    if (!["None", "True", "False", "lambda", "for", "in", "if", "else", "and", "or", "not"].includes(m[2])) out.add(m[2]);
  }
  return [...out];
}

/** The first line (1-based) that binds `name` (assignment, for / with target, def argument, import), or null. */
function assignmentLine(text, name) {
  if (!IDENT.test(name)) return null;
  const re = new RegExp(`^[ \\t]*(?:[\\w, ()]*\\b${name}\\b[\\w, ()]*(?::[^=\\n]+)?=[^=]|for\\s+[\\w, ()]*\\b${name}\\b[\\w, ()]*\\s+in\\b|with\\b.*\\bas\\s+${name}\\b|import\\b.*\\b${name}\\b)`);
  const i = text.split("\n").findIndex((ln) => re.test(ln));
  return i < 0 ? null : i + 1;
}

/** Where to evaluate `model` / `inputs`: after the cursor line and after every name they use is bound. */
function evalLine(text, cursorLine, ...exprs) {
  let line = cursorLine;
  for (const e of exprs) for (const n of namesIn(e || "")) {
    const a = assignmentLine(text, n);
    if (a && a > line && !/^\s*(import|from)\b/.test(text.split("\n")[a - 1])) line = a;
  }
  return line;
}

module.exports = { visualizeNames, visualizeCalls, closeParen, modelLines, moduleClasses, inputCandidates, namesIn, assignmentLine, evalLine };
