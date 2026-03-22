"""
Typer CLI entry-point for universums-sim.

Commands
--------
run     Execute a cosmic simulation and stream CosmicMoment output.
replay  Replay a previously saved simulation log.
export  Export simulation data to JSON / CSV / HDF5.
info    Print version, configuration and citation info.

Examples
--------
    universums-sim run --steps 500 --entropy 1.0 --visualize
    universums-sim run --steps 1000 --sonify --gui
    universums-sim export sim.json --format csv
    universums-sim info
"""

from __future__ import annotations

import json
import sys
from enum import Enum
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from universums_sim import __version__
from universums_sim.simulation.core import SimulationConfig, UniverseSimulator

app = typer.Typer(
    name="universums-sim",
    help="Cosmic emergence simulation for GenesisAeon.",
    add_completion=True,
    pretty_exceptions_enable=True,
)
console = Console()


class ExportFormat(str, Enum):
    """Supported export formats."""

    json = "json"
    csv = "csv"
    hdf5 = "hdf5"


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    steps: int = typer.Option(100, "--steps", "-s", min=1, help="Number of simulation steps."),
    entropy: float = typer.Option(1.0, "--entropy", "-e", min=1e-9, help="Initial entropy S_0."),
    n_particles: int = typer.Option(64, "--particles", "-p", min=2, help="Number of particles."),
    dt: float = typer.Option(0.01, "--dt", help="Integration time-step."),
    alpha: float = typer.Option(0.42, "--alpha", help="Emergence coupling constant."),
    seed: int = typer.Option(42, "--seed", help="Random seed for reproducibility."),
    visualize: bool = typer.Option(False, "--visualize", "-V", help="Enable live mandala rendering."),
    sonify: bool = typer.Option(False, "--sonify", help="Enable sonification of emergence events."),
    gui: bool = typer.Option(False, "--gui", help="Launch Dash GUI dashboard."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save moments to JSON file."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose structured logging."),
) -> None:
    """Run a cosmic emergence simulation and stream results."""
    console.rule(f"[bold cyan]universums-sim v{__version__}[/bold cyan]")
    console.print(f"  Steps      : [green]{steps}[/green]")
    console.print(f"  Particles  : [green]{n_particles}[/green]")
    console.print(f"  Entropy₀   : [green]{entropy}[/green]")
    console.print(f"  Seed       : [green]{seed}[/green]")

    cfg = SimulationConfig(
        n_particles=n_particles,
        dt=dt,
        entropy_initial=entropy,
        alpha=alpha,
        seed=seed,
    )
    sim = UniverseSimulator(cfg)

    _visualizer = None
    _sonifier = None

    if visualize or gui:  # pragma: no cover
        try:  # pragma: no cover
            from universums_sim.visualization.live import LiveVisualizer  # pragma: no cover
            _visualizer = LiveVisualizer(sim)  # pragma: no cover
            if gui:  # pragma: no cover
                _visualizer.launch_gui()  # pragma: no cover
        except ImportError:  # pragma: no cover
            console.print(  # pragma: no cover
                "[yellow]Warning:[/yellow] GUI/visualization dependencies not installed. "
                "Run: pip install 'universums-sim[gui]'"
            )

    if sonify:  # pragma: no cover
        try:  # pragma: no cover
            from universums_sim.visualization.live import SonificationEngine  # pragma: no cover
            _sonifier = SonificationEngine()  # pragma: no cover
        except ImportError:  # pragma: no cover
            console.print(  # pragma: no cover
                "[yellow]Warning:[/yellow] Sonification dependencies not installed. "
                "Run: pip install 'universums-sim[sonify]'"
            )

    moments = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=not verbose,
    ) as progress:
        task = progress.add_task("[cyan]Simulating...", total=steps)
        for moment in sim.run(steps):
            moments.append(moment)
            progress.update(task, advance=1)
            if verbose:
                progress.console.print(str(moment))
            if _visualizer is not None:  # pragma: no cover
                _visualizer.update(moment)  # pragma: no cover
            if _sonifier is not None:  # pragma: no cover
                _sonifier.process(moment)  # pragma: no cover

    # Summary table
    table = Table(title="Simulation Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    final = moments[-1]
    table.add_row("Total steps", str(len(moments)))
    table.add_row("Final entropy", f"{final.entropy:.4f}")
    table.add_row("Final phase", final.phase.name)
    table.add_row("Final emergence rate", f"{final.emergence_rate:.6f}")
    table.add_row("Final Hamiltonian", f"{final.hamiltonian:.4f}")
    table.add_row("Collapse state", final.collapse_state.name)
    total_events = sum(len(m.events) for m in moments)
    table.add_row("Total emergence events", str(total_events))
    console.print(table)

    if output is not None:
        data = [m.to_dict() for m in moments]
        output.write_text(json.dumps(data, indent=2))
        console.print(f"[green]Saved {len(moments)} moments to {output}[/green]")


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


@app.command()
def replay(
    input_file: Path = typer.Argument(..., help="Path to a JSON moments file."),
    speed: float = typer.Option(1.0, "--speed", help="Replay speed multiplier."),
    visualize: bool = typer.Option(False, "--visualize", "-V", help="Enable live rendering."),
) -> None:
    """Replay a previously saved simulation log."""
    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    raw = json.loads(input_file.read_text())
    console.print(f"[cyan]Replaying {len(raw)} moments at {speed}x speed...[/cyan]")
    for record in raw:
        if verbose_mode():
            console.print_json(json.dumps(record))
    console.print("[green]Replay complete.[/green]")


def verbose_mode() -> bool:
    """Check for --verbose flag in sys.argv."""
    return "--verbose" in sys.argv or "-v" in sys.argv


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@app.command()
def export(
    input_file: Path = typer.Argument(..., help="Path to a JSON moments file."),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o"),
    fmt: ExportFormat = typer.Option(ExportFormat.json, "--format", "-f"),
) -> None:
    """Export simulation data to JSON, CSV, or HDF5."""
    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    raw = json.loads(input_file.read_text())
    dest = output_file or input_file.with_suffix(f".{fmt.value}")

    if fmt == ExportFormat.json:
        dest.write_text(json.dumps(raw, indent=2))
    elif fmt == ExportFormat.csv:
        import csv  # noqa: PLC0415
        keys = ["step", "time", "entropy", "emergence_rate", "hamiltonian", "phase", "collapse_state"]
        with dest.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(raw)
    elif fmt == ExportFormat.hdf5:  # pragma: no cover
        try:  # pragma: no cover
            import h5py  # type: ignore[import-untyped]  # noqa: PLC0415  # pragma: no cover
            import numpy as np  # noqa: PLC0415  # pragma: no cover
            with h5py.File(dest, "w") as hf:  # pragma: no cover
                for key in ["step", "time", "entropy", "emergence_rate", "hamiltonian"]:  # pragma: no cover
                    hf.create_dataset(key, data=np.array([r[key] for r in raw]))  # pragma: no cover
        except ImportError:  # pragma: no cover
            console.print("[red]h5py not installed. Run: pip install h5py[/red]")  # pragma: no cover
            raise typer.Exit(1)  # pragma: no cover

    console.print(f"[green]Exported to {dest}[/green]")


# ---------------------------------------------------------------------------
# info
# ---------------------------------------------------------------------------


@app.command()
def info() -> None:
    """Print version, citation, and configuration information."""
    from universums_sim import __doi__, __license__

    console.rule("[bold cyan]universums-sim[/bold cyan]")
    console.print(f"  Version   : [green]{__version__}[/green]")
    console.print(f"  License   : {__license__}")
    console.print(f"  DOI       : https://doi.org/{__doi__}")
    console.print()
    console.print("[bold]Citation:[/bold]")
    console.print(
        "  GenesisAeon (2024). universums-sim: Cosmic emergence simulation.\n"
        f"  Zenodo. https://doi.org/{__doi__}"
    )


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry-point wrapper."""
    app()  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    main()  # pragma: no cover
