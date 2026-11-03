"""Read frozen CSV observations, including gzip-compressed files."""
from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

# Snapshot and candidate JSON cells can exceed the standard CSV limit.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
PAIR_KEY = ("seed", "scenario", "task_id")


def read_rows(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix.lower() == ".gz" else Path.open
    with opener(path, mode="rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
