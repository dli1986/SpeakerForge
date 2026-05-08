import pytest
import yaml
from pathlib import Path
from speakerforge.config import load_sources, load_pipeline, BilibiliSource, SourcesConfig, PipelineConfig


def test_load_sources_video(tmp_path):
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(yaml.dump({"sources": [
        {"speaker": "alice", "platform": "bilibili", "type": "video", "id": "BV1abc"}
    ]}))
    result = load_sources(str(cfg))
    assert isinstance(result, SourcesConfig)
    assert len(result.sources) == 1
    assert result.sources[0].speaker == "alice"
    assert result.sources[0].id == "BV1abc"


def test_load_sources_user_videos(tmp_path):
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(yaml.dump({"sources": [
        {"speaker": "bob", "platform": "bilibili", "type": "user_videos", "uid": "999", "limit": 5}
    ]}))
    result = load_sources(str(cfg))
    assert result.sources[0].uid == "999"
    assert result.sources[0].limit == 5


def test_load_pipeline_defaults(tmp_path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(yaml.dump({
        "auth": {"cookie_file": "~/.speakerforge/bili_cookies.json"},
        "stage3": {"min_duration": 0.8, "max_duration": 8.0, "min_silence_duration": 0.3},
    }))
    result = load_pipeline(str(cfg))
    assert result.auth["cookie_file"] == "~/.speakerforge/bili_cookies.json"
    assert result.stage3["min_duration"] == 0.8
    assert result.stage0 == {}
    assert result.stage1 == {}


def test_load_pipeline_empty_file(tmp_path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text("")
    result = load_pipeline(str(cfg))
    assert isinstance(result, PipelineConfig)
    assert result.auth == {}


def test_load_sources_unknown_key_raises(tmp_path):
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(yaml.dump({"sources": [
        {"speaker": "x", "platform": "bilibili", "type": "video", "id": "BV1", "typo_field": "oops"}
    ]}))
    with pytest.raises(ValueError, match="unknown fields"):
        load_sources(str(cfg))


def test_bilibili_source_optional_fields():
    src = BilibiliSource(speaker="x", platform="bilibili", type="video", id="BV1x")
    assert src.uid is None
    assert src.limit is None


def test_load_pipeline_unknown_keys_ignored(tmp_path):
    cfg = tmp_path / "pipeline.yaml"
    cfg.write_text(yaml.dump({
        "auth": {},
        "unknown_stage": {"foo": "bar"},
    }))
    # Should not raise
    result = load_pipeline(str(cfg))
    assert isinstance(result, PipelineConfig)


def test_load_sources_collection(tmp_path):
    cfg = tmp_path / "sources.yaml"
    cfg.write_text(yaml.dump({"sources": [
        {"speaker": "carol", "platform": "bilibili", "type": "collection", "id": "12345"}
    ]}))
    result = load_sources(str(cfg))
    assert result.sources[0].type == "collection"
    assert result.sources[0].id == "12345"
    assert result.sources[0].uid is None
