"""Settings loader from YAML config + environment variables."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

import yaml


CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
PROJECT_ROOT = Path(__file__).parent.parent.parent


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache
def load_settings() -> dict:
    settings_path = CONFIG_DIR / "settings.yaml"
    with open(settings_path) as f:
        settings = yaml.safe_load(f)

    # Override with env vars
    if api_key := os.environ.get("ANTHROPIC_API_KEY"):
        settings["anthropic_api_key"] = api_key
    if api_key := os.environ.get("OPENAI_API_KEY"):
        settings["openai_api_key"] = api_key

    # Resolve data_dir to absolute path
    settings["data_dir"] = str(PROJECT_ROOT / settings.get("data_dir", "data"))

    return settings


@lru_cache
def load_primitives() -> dict:
    with open(CONFIG_DIR / "primitives.yaml") as f:
        return yaml.safe_load(f)


@lru_cache
def load_domains() -> list[dict]:
    with open(CONFIG_DIR / "domains.yaml") as f:
        return yaml.safe_load(f)["domains"]


def load_brand_blocklist() -> list[str]:
    path = CONFIG_DIR / "brand_blocklist.txt"
    return [line.strip().lower() for line in path.read_text().splitlines() if line.strip()]


def load_prompt(name: str) -> str:
    path = CONFIG_DIR / "prompts" / f"{name}.txt"
    return path.read_text()


def get_all_primitive_names() -> list[str]:
    """Return flat list of all primitive names from taxonomy."""
    primitives = load_primitives()
    names = []
    for category in primitives.values():
        for p in category:
            names.append(p["name"])
    return names


def get_top10_primitives() -> list[str]:
    """Return names of the top-10 most important primitives."""
    primitives = load_primitives()
    names = []
    for category in primitives.values():
        for p in category:
            if p.get("top10"):
                names.append(p["name"])
    return names


def data_path(*parts: str) -> Path:
    settings = load_settings()
    p = Path(settings["data_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
