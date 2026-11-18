import subprocess
from pathlib import Path


def run_git(*args: str) -> None:
    subprocess.run(["git", *args], check=True)


def clone_sparse(
    repo: str,
    destination: Path,
    paths: list[str],
) -> None:
    if not (destination / ".git").exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        run_git(
            "clone",
            "--depth", "1",
            "--filter=blob:none",
            "--no-checkout",
            repo,
            str(destination),
        )

    run_git(
        "-C", str(destination),
        "sparse-checkout", "set", "--no-cone",
        *(f"/{path.lstrip('/')}" for path in paths),
    )
    run_git("-C", str(destination), "checkout", "HEAD")