const fs = require('fs');
const path = require('path');

const chunkSize = parseInt(process.argv[2] || '100', 10);
const sourceFile = path.join(__dirname, '..', 'docs', 'sigils', 'conversations.json');
const outDir = path.join(__dirname, '..', 'docs', 'sigils', 'fragments');

if (!fs.existsSync(sourceFile)) {
  console.error(`Source file not found: ${sourceFile}`);
  process.exit(1);
}

if (!fs.existsSync(outDir)) {
  fs.mkdirSync(outDir);
}

const data = JSON.parse(fs.readFileSync(sourceFile, 'utf8'));
let index = 0;
for (let i = 0; i < data.length; i += chunkSize) {
  const chunk = data.slice(i, i + chunkSize);
  const outPath = path.join(outDir, `fragment-${index}.json`);
  fs.writeFileSync(outPath, JSON.stringify(chunk, null, 2));
  console.log(`Wrote ${outPath}`);
  index++;
}
