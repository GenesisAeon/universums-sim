import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from packages.genesis_script import Engine

def test_step_runs():
    eng = Engine()
    assert eng.step() is None
