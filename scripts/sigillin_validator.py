#!/usr/bin/env python3
"""Validate Sigillin JSON/YAML files against the schema."""
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

REQUIRED_FIELDS = ["id", "title", "created_by", "timestamp"]


def load_file(path: Path):
    data = path.read_text(encoding='utf-8')
    if path.suffix in {'.yaml', '.yml'}:
        if yaml is None:
            raise RuntimeError('pyyaml is required for YAML files')
        return yaml.safe_load(data)
    return json.loads(data)


def validate_sigillin(obj: dict) -> bool:
    if not isinstance(obj, dict) or 'sigillin' not in obj:
        return False
    sig = obj['sigillin']
    if not isinstance(sig, dict):
        return False
    for field in REQUIRED_FIELDS:
        if field not in sig:
            return False
    return True


def main():
    if len(sys.argv) < 2:
        print('Usage: sigillin_validator.py <file>')
        sys.exit(1)
    path = Path(sys.argv[1])
    try:
        obj = load_file(path)
    except Exception as exc:
        print(f'Error reading {path}: {exc}')
        sys.exit(1)
    if validate_sigillin(obj):
        print('Valid Sigillin file')
        sys.exit(0)
    else:
        print('Invalid Sigillin file')
        sys.exit(2)


if __name__ == '__main__':
    main()
