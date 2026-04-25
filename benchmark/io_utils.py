import json
from pathlib import Path
import orjson
from typing import Iterable

from models import InputExample, CalibrationExample


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def read_jsonl(path: str) -> list[InputExample]:
    items: list[InputExample] = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(InputExample.model_validate(orjson.loads(line)))
    return items

def read_calibration_jsonl(path: str) -> list[CalibrationExample]:
    items: list[CalibrationExample] = []
    with open(path, "rb") as f:
        for line in f:
            if not line.strip():
                continue
            items.append(CalibrationExample.model_validate(orjson.loads(line)))
    return items


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        for row in rows:
            f.write(orjson.dumps(row))
            f.write(b"\n")


def write_json(path: str, obj: dict) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
