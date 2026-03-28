"""JSONL read/write utilities with Pydantic validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TypeVar, Type

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def read_jsonl(path: Path | str, model: Type[T]) -> Iterator[T]:
    """Stream-read a JSONL file, validating each line against a Pydantic model."""
    path = Path(path)
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            yield model.model_validate(data)


def read_jsonl_list(path: Path | str, model: Type[T]) -> list[T]:
    """Read all items from a JSONL file into a list."""
    return list(read_jsonl(path, model))


def write_jsonl(path: Path | str, items: list[BaseModel], append: bool = False) -> None:
    """Write a list of Pydantic models to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode) as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def append_jsonl(path: Path | str, item: BaseModel) -> None:
    """Append a single Pydantic model to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(item.model_dump_json() + "\n")


def count_lines(path: Path | str) -> int:
    """Count lines in a JSONL file."""
    path = Path(path)
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def load_processed_ids(path: Path | str, id_field: str = "id") -> set[str]:
    """Load the set of already-processed IDs from an output JSONL file."""
    path = Path(path)
    ids = set()
    if not path.exists():
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if id_field in data:
                ids.add(data[id_field])
    return ids
