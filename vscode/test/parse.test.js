const P = require("../parse");
const assert = require("assert");
const src = `import torch
import torch.nn as nn
import deepvis as dv
from deepvis import visualize as show

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)   # dv.visualize(not a call)
    def forward(self, x):
        return self.fc(x)

model = Net()
mlp = nn.Sequential(nn.Linear(4, 8), nn.ReLU())
x = torch.randn(3, 4)
t: torch.Tensor = torch.randint(0, 10, (3,))
dv.visualize(model, x,
             title="a (paren) ) in a string")
show(mlp, x)
# dv.visualize(model, x)
for step in range(3):
    loss = model(x).sum()
`;
assert.deepStrictEqual(P.visualizeNames(src).sort(), ["dv.visualize", "show"]);
const calls = P.visualizeCalls(src);
assert.deepStrictEqual(calls.map(c => [c.first, c.last]), [[17, 18], [19, 19]], JSON.stringify(calls));
assert.deepStrictEqual(P.modelLines(src), [{ name: "model", line: 13 }, { name: "mlp", line: 14 }]);
assert.deepStrictEqual(P.inputCandidates(src).map(c => c.name), ["x", "t"]);
assert.deepStrictEqual(P.namesIn("(x, t)"), ["x", "t"]);
assert.deepStrictEqual(P.namesIn("model.encoder"), ["model"]);
assert.strictEqual(P.assignmentLine(src, "x"), 15);
assert.strictEqual(P.assignmentLine(src, "step"), 21);
assert.strictEqual(P.evalLine(src, 13, "model", "x"), 15);
assert.strictEqual(P.evalLine(src, 16, "model", "x"), 16);
assert.deepStrictEqual(P.visualizeCalls("import deepvis\ndeepvis.visualize(m,\n  (x, '''\n)'''))\n").map(c => [c.first, c.last]), [[2, 4]]);
assert.deepStrictEqual(P.visualizeCalls("x = 1\n"), []);
console.log("parse ok");
// a notebook cell: the import is in another cell
assert.deepStrictEqual(P.visualizeCalls('dv.visualize(net, x)\n', "import torch\nimport deepvis as dv\n").map((c) => c.first), [1]);
assert.deepStrictEqual(P.visualizeCalls('vis(net, x)\n', "from deepvis import visualize as vis\n").map((c) => c.source), ["vis(net, x)"]);
assert.deepStrictEqual(P.visualizeCalls('dv.visualize(net, x)\n', "x = 1\n").length, 1);  // no import seen: the usual alias
assert.deepStrictEqual(P.visualizeCalls('dv.visualize(net, x)\n').length, 0);            // a plain file without the import
console.log("notebook context ok");
