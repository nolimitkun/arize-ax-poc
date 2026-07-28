"""Shared helpers for the poc/ scripts: console chrome, client, time windows."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from copilot.config import Settings, settings_or_exit  # noqa: E402

console = Console()

OUT_DIR = REPO_ROOT / ".out"


def header(step: str, title: str, docs: str) -> Settings:
    """Print the banner every script opens with, and load config."""
    console.print()
    console.print(
        Panel(
            f"[bold]{title}[/bold]\n[dim]{docs}[/dim]",
            title=f"[cyan]Arize AX POC · step {step}[/cyan]",
            border_style="cyan",
        )
    )
    settings = settings_or_exit()
    console.print(
        f"[dim]space={settings.arize_space_name}  project={settings.arize_project_name}[/dim]\n"
    )
    return settings


def arize_client(settings: Settings):
    from arize.client import ArizeClient

    return ArizeClient(api_key=settings.arize_api_key)


def window(hours: int = 24) -> tuple[datetime, datetime]:
    """A [start, end) window for span exports, padded into the future.

    The pad matters: ingestion timestamps can land slightly ahead of local
    clock, and an `end_time` of `now` silently drops the newest spans.
    """
    now = datetime.now(timezone.utc)
    return now - timedelta(hours=hours), now + timedelta(minutes=5)


def save(name: str, df) -> Path:
    """Persist an intermediate dataframe so later steps can pick it up."""
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / name
    df.to_parquet(path, index=False)
    console.print(f"[dim]wrote {path.relative_to(REPO_ROOT)} ({len(df)} rows)[/dim]")
    return path


def load(name: str):
    import pandas as pd

    path = OUT_DIR / name
    if not path.exists():
        console.print(
            f"[red]Missing {path.relative_to(REPO_ROOT)}.[/red] "
            "Run the earlier steps first (see `make help`)."
        )
        raise SystemExit(1)
    return pd.read_parquet(path)


def table(title: str, columns: list[str], rows: list[list[Any]]) -> None:
    t = Table(title=title, title_style="bold", header_style="bold cyan")
    for col in columns:
        t.add_column(col)
    for row in rows:
        t.add_row(*[str(c) for c in row])
    console.print(t)


def done(*next_steps: str) -> None:
    console.print()
    if next_steps:
        console.print("[bold green]Done.[/bold green] Next:")
        for step in next_steps:
            console.print(f"  • {step}")
    else:
        console.print("[bold green]Done.[/bold green]")
    console.print()


def look_at(*items: str) -> None:
    """What to go and check in the Arize UI after this step."""
    console.print("\n[bold yellow]Look at this in Arize:[/bold yellow]")
    for item in items:
        console.print(f"  → {item}")
