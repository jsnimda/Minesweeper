import os
import subprocess
from pathlib import Path

from .subprocess_pty import run_pty


def run_git(*args: str) -> None:
    if os.isatty(2):
        subprocess.run(["git", *args], check=True)
    else:
        run_pty(["git", *args], check=True)


def clone_repository(
    repo: str,
    destination: Path,
) -> None:
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)

        print(f"Cloning {repo}...", flush=True)

        run_git(
            "clone", "--quiet",
            "--depth", "1",
            "--filter=blob:none",
            "--no-checkout",
            repo,
            str(destination),
        )


def sparse_checkout(
    destination: Path,
    paths: list[str],
) -> None:
    run_git(
        "-C", str(destination),
        "sparse-checkout", "set", "--no-cone",
        *(f"/{path.lstrip('/')}" for path in paths),
    )

    run_git(
        "-C", str(destination),
        "checkout", "--quiet", "HEAD",
    )
