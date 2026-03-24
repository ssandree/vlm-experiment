from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(
    path: Path,
    data: Any,
    ensure_ascii: bool = False,
    indent: int = 2,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def write_json_bundle(
    base_dir: Path,
    payloads: dict[str, tuple[Any, bool] | Any],
) -> None:
    for filename, payload in payloads.items():
        if isinstance(payload, tuple):
            data, ensure_ascii = payload
            write_json(base_dir / filename, data, ensure_ascii=ensure_ascii)
        else:
            write_json(base_dir / filename, payload)

