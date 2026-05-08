import yaml
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BilibiliSource:
    speaker: str
    platform: str
    type: str  # video | collection | user_videos  # noqa: A003 (shadows builtin, required by schema)
    id: str | None = None
    uid: str | None = None
    limit: int | None = None


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
    """Load sources.yaml from path. Raises ValueError on malformed entries."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = set(BilibiliSource.__dataclass_fields__)
    sources = []
    for i, s in enumerate(data.get("sources", [])):
        unknown = set(s) - known
        if unknown:
            raise ValueError(f"sources[{i}] has unknown fields: {unknown}")
        sources.append(BilibiliSource(**s))
    return SourcesConfig(sources=sources)


def load_pipeline(path: str) -> PipelineConfig:
    """Load pipeline.yaml from path. Unknown top-level keys are silently ignored."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = set(PipelineConfig.__dataclass_fields__)
    return PipelineConfig(**{k: v for k, v in data.items() if k in known})
