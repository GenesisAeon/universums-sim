import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from packages.genesis_script import Engine

def test_step_runs():
    eng = Engine()
    assert eng.tick == 0
    eng.step()
    assert eng.tick == 1

class DummyEntity:
    def __init__(self):
        self.updated = False
    def update(self, engine):
        self.updated = True

def test_entity_update_called():
    eng = Engine()
    ent = DummyEntity()
    eng.add_entity(ent)
    eng.step()
    assert ent.updated

def test_add_entity():
    eng = Engine()
    ent = DummyEntity()
    eng.add_entity(ent)
    assert ent in eng.entities
