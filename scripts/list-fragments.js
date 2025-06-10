const fs = require('fs');
const path = require('path');

const fragDir = path.join(__dirname, '..', 'docs', 'sigils', 'fragments');
const progressFile = path.join(__dirname, '..', 'docs', 'sigils', 'conversations-progress.json');

const processed = fs.existsSync(progressFile)
  ? new Set(JSON.parse(fs.readFileSync(progressFile, 'utf8')))
  : new Set();

if (!fs.existsSync(fragDir)) {
  console.log('No fragments directory');
  process.exit(0);
}

const all = fs.readdirSync(fragDir).filter(f => f.endsWith('.json')).map(f => path.join('docs/sigils/fragments', f));

const pending = all.filter(f => !processed.has(f));
console.log('Total fragments:', all.length);
console.log('Processed:', processed.size);
console.log('Pending:', pending.length);
if (pending.length) {
  console.log('Pending files:');
  pending.forEach(f => console.log(' -', f));
}
