const fs = require('fs');
const path = require('path');

const ignoreDirs = new Set(['.git', 'node_modules', 'docs/sigils']);
const todos = [];

function walk(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dir, entry.name);
    if (ignoreDirs.has(entry.name) || ignoreDirs.has(dir)) continue;
    if (entry.isDirectory()) {
      walk(full);
    } else if (entry.isFile()) {
      const content = fs.readFileSync(full, 'utf8');
      const lines = content.split(/\r?\n/);
      lines.forEach((line, idx) => {
        if (line.includes('TODO')) {
          todos.push({ file: full, line: idx + 1, text: line.trim() });
        }
      });
    }
  }
}

walk('.');

const outLines = ['todos:'];
for (const t of todos) {
  outLines.push(`  - file: ${t.file}`);
  outLines.push(`    line: ${t.line}`);
  outLines.push(`    text: "${t.text.replace(/"/g, '\\"')}"`);
}

const outPath = path.join('docs', 'sigils', 'generated-todo.sigil.yaml');
fs.writeFileSync(outPath, outLines.join('\n'));
console.log(`Wrote ${outPath} with ${todos.length} entries`);
