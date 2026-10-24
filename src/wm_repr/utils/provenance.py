"""Identify the code revision of a git checkout."""
from pathlib import Path
import subprocess


def source_revision():
    """Return the checkout's HEAD commit, or None outside a git checkout."""
    checkout = Path(__file__).resolve().parents[3]
    if (checkout / '.git').exists():
        try:
            return subprocess.check_output(
                ['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                text=True, stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return None
