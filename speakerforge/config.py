from __future__ import annotations
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class BilibiliSource:
    speaker: str
    platform: str
    type: str  # video | collection | user_videos
    id: Optional[str] = None
    uid: Optional[str] = None
    limit: Optional[int] = None


@dataclass
class SourcesConfig:
    sources: list[BilibiliSource]


@dataclass
class PipelineConfig:
    auth: dict = field(default_factory=dict)
    stage0: dict = field(default_factory=dict)
    stage1: dict = field(default_factory=dict)
    stage2: dict = field(default_factory=dict)
    stage3: dict = field(default_factory=dict)
    stage4: dict = field(default_factory=dict)
    stage5: dict = field(default_factory=dict)
    stage6: dict = field(default_factory=dict)
    stage7: dict = field(default_factory=dict)
    stage8: dict = field(default_factory=dict)


def load_sources(path: str) -> SourcesConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sources = [BilibiliSource(**s) for s in data["sources"]]
    return SourcesConfig(sources=sources)


def load_pipeline(path: str) -> PipelineConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    known = set(PipelineConfig.__dataclass_fields__)
    filtered = {k: v for k, v in data.items() if k in known}
    return PipelineConfig(**filtered)
