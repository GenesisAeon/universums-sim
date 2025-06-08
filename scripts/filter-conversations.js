const fs = require('fs');
const path = require('path');

const keyword = process.argv[2];
if (!keyword) {
  console.error('Usage: node filter-conversations.js <keyword>');
  process.exit(1);
}
const sourceFile = path.join(__dirname, '..', 'docs', 'sigils', 'conversations.json');
if (!fs.existsSync(sourceFile)) {
  console.error(`Source file not found: ${sourceFile}`);
  process.exit(1);
}
const data = JSON.parse(fs.readFileSync(sourceFile, 'utf8'));
const filtered = data.filter(entry => JSON.stringify(entry).includes(keyword));
const outPath = path.join(__dirname, '..', 'docs', 'sigils', `conversations-filter-${keyword}.json`);
fs.writeFileSync(outPath, JSON.stringify(filtered, null, 2));
console.log(`Wrote ${outPath}`);
