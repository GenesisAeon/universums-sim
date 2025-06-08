const fs = require('fs');
const path = require('path');

const fragmentPath = process.argv[2];
if (!fragmentPath) {
  console.error('Usage: node mark-fragment.js <fragment-path>');
  process.exit(1);
}
const progressFile = path.join(__dirname, '..', 'docs', 'sigils', 'conversations-progress.json');
let progress = [];
if (fs.existsSync(progressFile)) {
  progress = JSON.parse(fs.readFileSync(progressFile, 'utf8'));
}
if (!progress.includes(fragmentPath)) {
  progress.push(fragmentPath);
  fs.writeFileSync(progressFile, JSON.stringify(progress, null, 2));
  console.log(`Marked ${fragmentPath}`);
} else {
  console.log(`${fragmentPath} already marked`);
}
