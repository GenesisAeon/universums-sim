"""GenesisScript Basismodul."""

class Engine:
    def __init__(self):
        self.entities = []
        self.tick = 0

    def add_entity(self, entity):
        """Fügt eine Entität zur Simulation hinzu."""
        if entity is not None:
            self.entities.append(entity)

    def step(self):
        """Führt einen Simulationsschritt aus."""
        self.tick += 1
        for entity in self.entities:
            update = getattr(entity, "update", None)
            if callable(update):
                update(self)
