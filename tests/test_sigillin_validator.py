import os
import sys
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.sigillin_validator import validate_sigillin, load_file


def test_validate_sigillin_valid_json(tmp_path):
    data = {
        "sigillin": {
            "id": "sig-test",
            "title": "Test",
            "created_by": "tester",
            "timestamp": "2025-06-10T00:00:00Z"
        }
    }
    file = tmp_path / "test.json"
    file.write_text(json.dumps(data))
    loaded = load_file(file)
    assert validate_sigillin(loaded)


def test_validate_sigillin_invalid_missing_field(tmp_path):
    data = {"sigillin": {"id": "x"}}
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(data))
    loaded = load_file(file)
    assert not validate_sigillin(loaded)
