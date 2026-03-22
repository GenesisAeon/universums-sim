# Changelog

## 0.1.0 (2024-03-22)

### Added

- `UniverseSimulator` with symplectic leapfrog integrator
- `UnifiedLagrangian`: kinetic, gravitational, scalar, entropic, topological terms
- `EmergenceEngine` with Poisson-process event firing
- `CosmicMoment` immutable event snapshots
- `CollapseState` virial-ratio collapse detection
- `EntropyGovernor` with UTAC-compatible policy
- `IntegrationRegistry` for optional full-stack packages
- `LiveVisualizer`: MandalaRenderer, Emergence3D, SonificationEngine, DashDashboard
- Typer CLI: run / replay / export / info commands
- >600 pytest tests, >99% coverage
- mkdocs --strict documentation with KaTeX formulas
- Zenodo-ready `.zenodo.json`
- GitHub Actions CI/CD + PyPI release workflow
