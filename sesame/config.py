from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SesameConfig:
    dataset: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    lora: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    eval: dict = field(default_factory=dict)


def load_sesame_config(path: str) -> SesameConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = set(SesameConfig.__dataclass_fields__)
    return SesameConfig(**{k: v for k, v in data.items() if k in known})
