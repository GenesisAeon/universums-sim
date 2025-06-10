#!/bin/sh
# Synchronize directories based on codex_sync_manifest.yaml
set -e
manifest="codex_sync_manifest.yaml"
if [ ! -f "$manifest" ]; then
  echo "Manifest not found: $manifest" >&2
  exit 1
fi

src=""
dst=""
while IFS= read -r line; do
  case "$line" in
    "  - id:"*)
      if [ -n "$src" ] && [ -n "$dst" ]; then
        mkdir -p "$dst"
        rsync -a "$src" "$dst"
      fi
      src=""; dst="" ;;
    "    quelle:"*)
      src="$(echo "$line" | cut -d: -f2- | tr -d ' ')" ;;
    "    ziel:"*)
      dst="$(echo "$line" | cut -d: -f2- | tr -d ' ')" ;;
  esac
done < "$manifest"

if [ -n "$src" ] && [ -n "$dst" ]; then
  mkdir -p "$dst"
  rsync -a "$src" "$dst"
fi
