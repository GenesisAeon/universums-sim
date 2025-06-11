#!/bin/sh
# Generate agents_chain.svg from agents_chain.mmd using mermaid-cli if available
if command -v mmdc >/dev/null 2>&1; then
  mmdc -i agents_chain.mmd -o agents_chain.svg
else
  if npx --yes @mermaid-js/mermaid-cli -h >/dev/null 2>&1; then
    npx --yes @mermaid-js/mermaid-cli -i agents_chain.mmd -o agents_chain.svg
  else
    echo "Mermaid CLI not available. Install @mermaid-js/mermaid-cli to generate SVG." >&2
    exit 1
  fi
fi
