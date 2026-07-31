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
        f"[dim]space={settings.arize_space_name}  project={settings.arize_project_name}  "
        f"region={settings.arize_region or 'default (US)'}[/dim]\n"
    )
    return settings


def arize_client(settings: Settings):
    """The platform client, region-matched to the trace exporter.

    An EU space needs region here as well as a EU collector URL in tracing.py;
    fixing only the latter leaves every span export and dataset call hitting
    the default region with a key that isn't valid there.
    """
    from copilot.config import platform_client

    return platform_client(settings)


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


def mcnemar_p(only_a_passed: int, only_b_passed: int) -> float:
    """Two-sided exact McNemar p-value for paired pass/fail outcomes.

    Used wherever two variants are measured on the *same* rows -- poc/08's v1
    against v2, poc/06b's judge against its aligned self. Because the rows are
    shared, the only informative ones are those where the two disagree. Under
    the null those disagreements split 50/50, which makes this a plain two-sided
    binomial test on `only_b_passed` out of the total disagreement. Exact rather
    than the chi-square approximation because these counts are small.
    """
    from math import comb

    n = only_a_passed + only_b_passed
    if n == 0:
        return 1.0
    k = min(only_a_passed, only_b_passed)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


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
