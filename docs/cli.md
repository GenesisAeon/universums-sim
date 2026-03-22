# CLI Reference

The `universums-sim` CLI is built with [Typer](https://typer.tiangolo.com/).

## run

```bash
universums-sim run --steps 500 --particles 64 --entropy 1.0 --visualize --sonify --gui
```

## export

```bash
universums-sim export sim.json --format csv
universums-sim export sim.json --format hdf5
```

## replay

```bash
universums-sim replay sim.json --speed 2.0
```

## info

```bash
universums-sim info
```
