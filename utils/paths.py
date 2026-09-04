"""Project paths independent of the caller's working directory."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATABASE = Path(os.getenv('TP_DATABASE_DIR', str(ROOT / 'database'))).expanduser().resolve()
FROZEN = ROOT / 'results' / 'frozen'


def seed_path(kind):
    method = {'direct': 'direct', 'program': 'program_v23_best'}[kind]
    explicit = os.getenv(f'{kind.upper()}_SUBMISSION_FILE')
    if explicit:
        return Path(explicit).expanduser().resolve()
    return FROZEN / 'predictions' / f'{method}.jsonl'
