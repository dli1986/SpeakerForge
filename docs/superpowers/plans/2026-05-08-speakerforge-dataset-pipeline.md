# SpeakerForge Dataset Pipeline (Stages 0–7) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular CLI pipeline that downloads Bilibili videos from a config-driven whitelist and converts them into a clean, LJSpeech-format speech dataset ready for CosyVoice 2 LoRA fine-tuning.

**Architecture:** A Python package `speakerforge` with a Click CLI entry point (`python -m speakerforge <stage>`). Each stage reads from and writes to a well-defined directory under `processed/<speaker>/` and `dataset/<speaker>/`, making every stage independently re-runnable. Bilibili resolution uses `bilibili_api`; actual downloads use `yt-dlp`. Audio processing uses Demucs → silero-vad → WhisperX in sequence.

**Tech Stack:** Python 3.10, Click, bilibili-api-python, yt-dlp, ffmpeg-python, demucs, whisperx, silero-vad, soundfile, librosa, pydub, pyyaml, pytest

---

## File Map

| File | Responsibility |
|---|---|
| `speakerforge/__init__.py` | Package marker |
| `speakerforge/__main__.py` | `python -m speakerforge` entry |
| `speakerforge/cli.py` | Click group + all stage subcommands |
| `speakerforge/bili_auth.py` | Load/save/refresh bilibili_api Credential from JSON |
| `speakerforge/config.py` | Load `sources.yaml` and `pipeline.yaml` into dataclasses |
| `speakerforge/stages/stage0_acquire.py` | Resolve sources → BV list; download via yt-dlp |
| `speakerforge/stages/stage1_extract.py` | ffmpeg: video → mono 24kHz WAV |
| `speakerforge/stages/stage2_demucs.py` | htdemucs_ft: WAV → vocals-only WAV |
| `speakerforge/stages/stage3_vad.py` | silero-vad: long WAV → short segment WAVs |
| `speakerforge/stages/stage4_align.py` | WhisperX: segments → aligned JSON |
| `speakerforge/stages/stage5_filter.py` | Filter segments by duration/confidence/text length |
| `speakerforge/stages/stage6_normalize.py` | RMS normalize + silence trim |
| `speakerforge/stages/stage7_package.py` | Package into `dataset/<speaker>/audio/` + `metadata.csv` |
| `tests/conftest.py` | Shared fixtures (tmp dirs, sample WAV factory) |
| `tests/test_config.py` | Config loading tests |
| `tests/test_bili_auth.py` | Credential load/save roundtrip tests |
| `tests/test_stage1.py` | ffmpeg extraction tests (with mock) |
| `tests/test_stage3.py` | VAD segmentation tests |
| `tests/test_stage5.py` | Filter logic tests (pure Python, no model needed) |
| `tests/test_stage7.py` | Dataset packaging tests |
| `sources.yaml.example` | Source whitelist template |
| `pipeline.yaml` | Per-stage hyperparameters |
| `requirements.txt` | All pip dependencies |
| `pyproject.toml` | Package metadata + entry point |

---

## Task 1: Project Scaffolding

**Files:**
- Create: `speakerforge/__init__.py`
- Create: `speakerforge/__main__.py`
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `sources.yaml.example`
- Create: `pipeline.yaml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "speakerforge"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "click>=8.0",
    "pyyaml>=6.0",
    "tqdm>=4.0",
]

[project.scripts]
speakerforge = "speakerforge.cli:cli"

[tool.setuptools.packages.find]
where = ["."]
include = ["speakerforge*"]
```

- [ ] **Step 2: Write `speakerforge/__init__.py`**

```python
__version__ = "0.1.0"
```

- [ ] **Step 3: Write `speakerforge/__main__.py`**

```python
from speakerforge.cli import cli

if __name__ == "__main__":
    cli()
```

- [ ] **Step 4: Write `requirements.txt`**

```
# Bilibili
bilibili-api-python>=17.0.0
yt-dlp>=2024.1.1

# Audio processing
ffmpeg-python>=0.2.0
demucs>=4.0.0
whisperx>=3.1.0
silero-vad>=5.1
soundfile>=0.12.0
librosa>=0.10.0
pydub>=0.25.0

# ML
torch>=2.0.0
torchaudio>=2.0.0

# Utils
click>=8.0
pyyaml>=6.0
tqdm>=4.0
pandas>=2.0.0
resemblyzer>=0.1.1.dev0

# Dev
pytest>=7.0
pytest-mock>=3.0
```

- [ ] **Step 5: Write `sources.yaml.example`**

```yaml
sources:
  - speaker: "up_ai_report"
    platform: bilibili
    type: video
    id: "BV1xxxx"

  - speaker: "up_ai_report"
    platform: bilibili
    type: collection
    id: "12345"

  - speaker: "up_voice_clean"
    platform: bilibili
    type: user_videos
    uid: "123456"
    limit: 10
```

- [ ] **Step 6: Write `pipeline.yaml`**

```yaml
auth:
  cookie_file: "~/.speakerforge/bili_cookies.json"
  # override: "../Subtitle/cache/bili_cookies.json"

stage1:
  sample_rate: 24000

stage2:
  model: htdemucs_ft

stage3:
  min_duration: 0.8
  max_duration: 8.0
  min_silence_duration: 0.3

stage4:
  model: large-v3
  language: zh
  batch_size: 16
  # use "medium" for CPU-only (no GPU with >=8GB VRAM)

stage5:
  min_duration: 1.0
  max_duration: 6.0
  min_confidence: 0.6
  min_text_chars: 4

stage6:
  target_rms_dbfs: -20
  silence_threshold_dbfs: -40

stage8:
  base_model: cosyvoice2
  cosyvoice2_repo: "/path/to/CosyVoice"
  lora_rank: 16
  epochs: 10
  batch_size: 4
```

- [ ] **Step 7: Install package in dev mode**

```bash
cd c:/Users/dli/Projects/MyTest/SpeakerForge
pip install -e .
```

Expected: `Successfully installed speakerforge-0.1.0`

- [ ] **Step 8: Install required packages not yet in venv**

```bash
pip install demucs whisperx silero-vad soundfile librosa pydub pandas resemblyzer pytest pytest-mock
```

Expected: All packages install successfully. Note: `whisperx` may require `pip install whisperx` from PyPI or GitHub. If PyPI fails: `pip install git+https://github.com/m-bain/whisperX.git`

- [ ] **Step 9: Commit**

```bash
git init  # only if not already a git repo
git add pyproject.toml requirements.txt pipeline.yaml sources.yaml.example speakerforge/__init__.py speakerforge/__main__.py
git commit -m "feat: scaffold SpeakerForge project"
```

---

## Task 2: Config Module

**Files:**
- Create: `speakerforge/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_config.py
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


def test_bilibili_source_optional_fields():
    src = BilibiliSource(speaker="x", platform="bilibili", type="video", id="BV1x")
    assert src.uid is None
    assert src.limit is None
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
cd c:/Users/dli/Projects/MyTest/SpeakerForge
pytest tests/test_config.py -v
```

Expected: `ImportError: cannot import name 'load_sources'`

- [ ] **Step 3: Write `speakerforge/config.py`**

```python
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
    stage1: dict = field(default_factory=dict)
    stage2: dict = field(default_factory=dict)
    stage3: dict = field(default_factory=dict)
    stage4: dict = field(default_factory=dict)
    stage5: dict = field(default_factory=dict)
    stage6: dict = field(default_factory=dict)
    stage8: dict = field(default_factory=dict)


def load_sources(path: str) -> SourcesConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sources = [BilibiliSource(**s) for s in data["sources"]]
    return SourcesConfig(sources=sources)


def load_pipeline(path: str) -> PipelineConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return PipelineConfig(**{k: v for k, v in data.items() if k in PipelineConfig.__dataclass_fields__})
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
pytest tests/test_config.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add speakerforge/config.py tests/test_config.py
git commit -m "feat: add config loader for sources.yaml and pipeline.yaml"
```

---

## Task 3: Bilibili Auth Module

**Files:**
- Create: `speakerforge/bili_auth.py`
- Create: `tests/test_bili_auth.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_bili_auth.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from speakerforge.bili_auth import load_credential, save_credential, get_or_login


SAMPLE_CRED_DATA = {
    "sessdata": "sess123",
    "bili_jct": "jct456",
    "buvid3": "buv789",
    "dedeuserid": "uid000",
    "ac_time_value": "act111",
}


def test_load_credential_returns_none_if_missing(tmp_path):
    from speakerforge.bili_auth import load_credential
    result = load_credential(str(tmp_path / "missing.json"))
    assert result is None


def test_load_credential_returns_credential(tmp_path):
    cookie_file = tmp_path / "creds.json"
    cookie_file.write_text(json.dumps(SAMPLE_CRED_DATA))
    with patch("speakerforge.bili_auth.Credential") as MockCred:
        MockCred.return_value = MagicMock()
        result = load_credential(str(cookie_file))
        MockCred.assert_called_once_with(
            sessdata="sess123",
            bili_jct="jct456",
            buvid3="buv789",
            dedeuserid="uid000",
            ac_time_value="act111",
        )
        assert result is not None


def test_save_credential_writes_json(tmp_path):
    cookie_file = tmp_path / "creds.json"
    mock_cred = MagicMock()
    mock_cred.sessdata = "s"
    mock_cred.bili_jct = "j"
    mock_cred.buvid3 = "b"
    mock_cred.dedeuserid = "d"
    mock_cred.ac_time_value = "a"
    save_credential(mock_cred, str(cookie_file))
    data = json.loads(cookie_file.read_text())
    assert data["sessdata"] == "s"
    assert data["bili_jct"] == "j"


def test_get_or_login_returns_existing(tmp_path):
    cookie_file = tmp_path / "creds.json"
    cookie_file.write_text(json.dumps(SAMPLE_CRED_DATA))
    with patch("speakerforge.bili_auth.Credential") as MockCred:
        mock_instance = MagicMock()
        MockCred.return_value = mock_instance
        result = get_or_login(str(cookie_file), login_if_missing=False)
        assert result is mock_instance
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest tests/test_bili_auth.py -v
```

Expected: `ImportError: cannot import name 'load_credential'`

- [ ] **Step 3: Write `speakerforge/bili_auth.py`**

Adapted from `Subtitle/uploader/bilibili_uploader.py` — credential management only, no upload logic.

```python
from __future__ import annotations
import asyncio
import json
import threading
from pathlib import Path
from typing import Optional

from bilibili_api.login_v2 import Credential


def load_credential(cookie_file: str) -> Optional[Credential]:
    path = Path(cookie_file).expanduser()
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return Credential(
        sessdata=data["sessdata"],
        bili_jct=data["bili_jct"],
        buvid3=data["buvid3"],
        dedeuserid=data["dedeuserid"],
        ac_time_value=data["ac_time_value"],
    )


def save_credential(cred: Credential, cookie_file: str) -> None:
    path = Path(cookie_file).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "sessdata": cred.sessdata,
        "bili_jct": cred.bili_jct,
        "buvid3": cred.buvid3,
        "dedeuserid": cred.dedeuserid,
        "ac_time_value": cred.ac_time_value,
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def get_or_login(cookie_file: str, login_if_missing: bool = True) -> Credential:
    cred = load_credential(cookie_file)
    if cred is not None:
        return cred
    if not login_if_missing:
        raise RuntimeError(
            f"No credential found at {cookie_file}. "
            "Run: python -m speakerforge login"
        )
    return _qr_login(cookie_file)


def _qr_login(cookie_file: str) -> Credential:
    """Interactive QR-code login. Reuses persistent event loop to avoid bilibili_api internal loop conflicts."""
    from bilibili_api.login_v2 import QrCodeLogin, QrCodeLoginChannel, QrCodeLoginState

    loop = asyncio.new_event_loop()

    async def _do_login():
        login = QrCodeLogin(platform=QrCodeLoginChannel.WEB)
        await login.start()
        print(login.get_qrcode_terminal())
        print("Scan the QR code with Bilibili app, then confirm login.")
        while True:
            state = await login.check_state()
            if state == QrCodeLoginState.DONE:
                return login.get_credential()
            if state == QrCodeLoginState.TIMEOUT:
                raise RuntimeError("QR code timed out. Please retry.")
            await asyncio.sleep(2)

    try:
        cred = loop.run_until_complete(_do_login())
    finally:
        loop.close()

    save_credential(cred, cookie_file)
    print(f"Credentials saved to {cookie_file}")
    return cred
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
pytest tests/test_bili_auth.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add speakerforge/bili_auth.py tests/test_bili_auth.py
git commit -m "feat: add bili_auth credential load/save/login"
```

---

## Task 4: CLI Skeleton + `login` Subcommand

**Files:**
- Create: `speakerforge/cli.py`
- Create: `speakerforge/stages/__init__.py`

- [ ] **Step 1: Write `speakerforge/stages/__init__.py`**

```python
```
(empty file)

- [ ] **Step 2: Write `speakerforge/cli.py`**

```python
import click
from pathlib import Path

DEFAULT_PIPELINE = "pipeline.yaml"
DEFAULT_SOURCES = "sources.yaml"


@click.group()
def cli():
    """SpeakerForge: Bilibili → speech dataset pipeline."""
    pass


@cli.command()
@click.option("--cookie-file", default="~/.speakerforge/bili_cookies.json",
              show_default=True, help="Path to bili_cookies.json")
def login(cookie_file):
    """Interactive QR-code login to Bilibili (one-time setup)."""
    from speakerforge.bili_auth import _qr_login
    _qr_login(cookie_file)


@cli.command()
@click.option("--speaker", required=True, help="Speaker name from sources.yaml")
@click.option("--sources", default=DEFAULT_SOURCES, show_default=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--dry-run", is_flag=True)
def stage0(speaker, sources, config, dry_run):
    """Stage 0: Resolve Bilibili sources and download videos."""
    from speakerforge.stages.stage0_acquire import run
    from speakerforge.config import load_sources, load_pipeline
    run(load_sources(sources), load_pipeline(config), speaker=speaker, dry_run=dry_run)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage1(speaker, config):
    """Stage 1: Extract audio from downloaded videos."""
    from speakerforge.stages.stage1_extract import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage2(speaker, config):
    """Stage 2: Vocal isolation with Demucs."""
    from speakerforge.stages.stage2_demucs import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage3(speaker, config):
    """Stage 3: VAD segmentation with silero-vad."""
    from speakerforge.stages.stage3_vad import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage4(speaker, config):
    """Stage 4: WhisperX transcription + forced alignment."""
    from speakerforge.stages.stage4_align import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage5(speaker, config):
    """Stage 5: Quality filtering."""
    from speakerforge.stages.stage5_filter import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
def stage6(speaker, config):
    """Stage 6: RMS normalization + silence trimming."""
    from speakerforge.stages.stage6_normalize import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--version", default="B", type=click.Choice(["A", "B"]),
              help="A=baseline (no Demucs/VAD), B=full pipeline")
def stage7(speaker, config, version):
    """Stage 7: Package into LJSpeech-format dataset."""
    from speakerforge.stages.stage7_package import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker, version=version)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--sources", default=DEFAULT_SOURCES, show_default=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--stages", default="0-7", show_default=True,
              help="Stage range, e.g. '0-7' or '2-5'")
@click.option("--version", default="B", type=click.Choice(["A", "B"]))
def run(speaker, sources, config, stages, version):
    """Run a range of stages in sequence."""
    from speakerforge.config import load_sources, load_pipeline
    src_cfg = load_sources(sources)
    pipe_cfg = load_pipeline(config)

    start, end = (int(x) for x in stages.split("-"))
    stage_fns = {
        0: lambda: __import__("speakerforge.stages.stage0_acquire", fromlist=["run"]).run(src_cfg, pipe_cfg, speaker=speaker),
        1: lambda: __import__("speakerforge.stages.stage1_extract", fromlist=["run"]).run(pipe_cfg, speaker=speaker),
        2: lambda: __import__("speakerforge.stages.stage2_demucs", fromlist=["run"]).run(pipe_cfg, speaker=speaker),
        3: lambda: __import__("speakerforge.stages.stage3_vad", fromlist=["run"]).run(pipe_cfg, speaker=speaker),
        4: lambda: __import__("speakerforge.stages.stage4_align", fromlist=["run"]).run(pipe_cfg, speaker=speaker),
        5: lambda: __import__("speakerforge.stages.stage5_filter", fromlist=["run"]).run(pipe_cfg, speaker=speaker),
        6: lambda: __import__("speakerforge.stages.stage6_normalize", fromlist=["run"]).run(pipe_cfg, speaker=speaker),
        7: lambda: __import__("speakerforge.stages.stage7_package", fromlist=["run"]).run(pipe_cfg, speaker=speaker, version=version),
    }
    for i in range(start, end + 1):
        click.echo(f"\n{'='*50}\nRunning stage {i}...\n{'='*50}")
        stage_fns[i]()


@cli.command()
@click.option("--speaker", required=True)
def stats(speaker):
    """Show dataset statistics for a speaker."""
    import pandas as pd
    from pathlib import Path
    for version in ["A", "B"]:
        csv = Path(f"dataset/{speaker}_version{version}/metadata.csv")
        if csv.exists():
            df = pd.read_csv(csv)
            total = df["duration"].sum()
            click.echo(f"\nVersion {version}: {len(df)} segments, {total/60:.1f} min total")
            click.echo(df["duration"].describe().to_string())
```

- [ ] **Step 3: Verify CLI entry point works**

```bash
python -m speakerforge --help
```

Expected output lists: `login`, `stage0` through `stage7`, `run`, `stats`

- [ ] **Step 4: Commit**

```bash
git add speakerforge/cli.py speakerforge/stages/__init__.py
git commit -m "feat: add CLI skeleton with all stage subcommands"
```

---

## Task 5: Stage 0 — Bilibili Source Acquisition

**Files:**
- Create: `speakerforge/stages/stage0_acquire.py`

- [ ] **Step 1: Write `speakerforge/stages/stage0_acquire.py`**

```python
from __future__ import annotations
import asyncio
import json
import subprocess
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from speakerforge.bili_auth import get_or_login
from speakerforge.config import BilibiliSource, PipelineConfig, SourcesConfig

RAW_DIR = Path("raw_sources/bilibili")


def run(sources_cfg: SourcesConfig, pipeline_cfg: PipelineConfig,
        speaker: Optional[str] = None, dry_run: bool = False) -> None:
    cookie_file = pipeline_cfg.auth.get("cookie_file", "~/.speakerforge/bili_cookies.json")
    cred = get_or_login(cookie_file)

    filtered = [s for s in sources_cfg.sources
                if speaker is None or s.speaker == speaker]

    loop = asyncio.new_event_loop()
    try:
        for source in filtered:
            bv_ids = loop.run_until_complete(_resolve(source, cred))
            out_dir = RAW_DIR / source.speaker
            out_dir.mkdir(parents=True, exist_ok=True)
            _update_index(out_dir, source, bv_ids)
            if dry_run:
                print(f"[dry-run] {source.speaker}: {bv_ids}")
                continue
            for bv in tqdm(bv_ids, desc=f"Downloading {source.speaker}"):
                _download(bv, out_dir, cookie_file)
    finally:
        loop.close()


async def _resolve(source: BilibiliSource, cred) -> list[str]:
    if source.type == "video":
        return [source.id]
    if source.type in ("collection", "playlist"):
        return await _resolve_collection(source.id, cred)
    if source.type == "user_videos":
        return await _resolve_user_videos(source.uid, source.limit or 10, cred)
    raise ValueError(f"Unknown source type: {source.type}")


async def _resolve_collection(collection_id: str, cred) -> list[str]:
    from bilibili_api.channel_series import ChannelSeries
    series = ChannelSeries(series_id=int(collection_id), credential=cred)
    videos = await series.get_videos()
    return [v["bvid"] for v in videos.get("archives", [])]


async def _resolve_user_videos(uid: str, limit: int, cred) -> list[str]:
    from bilibili_api.user import User
    user = User(uid=int(uid), credential=cred)
    result = await user.get_videos(ps=limit)
    return [v["bvid"] for v in result.get("list", {}).get("vlist", [])][:limit]


def _download(bv_id: str, out_dir: Path, cookie_file: str) -> None:
    out_path = out_dir / f"{bv_id}.mp4"
    if out_path.exists():
        print(f"  Skip (exists): {bv_id}")
        return
    cookie_path = str(Path(cookie_file).expanduser())
    url = f"https://www.bilibili.com/video/{bv_id}/"
    cmd = [
        "yt-dlp",
        "--cookies", cookie_path,
        "--format", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]",
        "--merge-output-format", "mp4",
        "--limit-rate", "1.5M",
        "--retries", "3",
        "--output", str(out_dir / f"{bv_id}.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)


def _update_index(out_dir: Path, source: BilibiliSource, bv_ids: list[str]) -> None:
    index_path = out_dir / "source_index.json"
    index = {}
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
    for bv in bv_ids:
        if bv not in index:
            index[bv] = {"speaker": source.speaker, "status": "pending", "duration": None}
    index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from speakerforge.stages.stage0_acquire import run; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Test dry-run with example sources**

Copy `sources.yaml.example` to `sources.yaml`, add a real BV ID you own/can access, then:

```bash
python -m speakerforge stage0 --speaker up_ai_report --dry-run
```

Expected: prints BV IDs without downloading.

- [ ] **Step 4: Commit**

```bash
git add speakerforge/stages/stage0_acquire.py
git commit -m "feat: stage0 Bilibili acquisition with bilibili_api resolve + yt-dlp download"
```

---

## Task 6: Stage 1 — Audio Extraction

**Files:**
- Create: `speakerforge/stages/stage1_extract.py`
- Create: `tests/test_stage1.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import numpy as np
import pytest
import soundfile as sf
from pathlib import Path


@pytest.fixture
def make_wav(tmp_path):
    """Factory: create a synthetic WAV file at given path."""
    def _make(filename: str, duration: float = 3.0, sr: int = 24000) -> Path:
        path = tmp_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        samples = np.random.randn(int(duration * sr)).astype(np.float32) * 0.1
        sf.write(str(path), samples, sr)
        return path
    return _make
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_stage1.py
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from speakerforge.stages.stage1_extract import extract_audio, run
from speakerforge.config import PipelineConfig


def test_extract_audio_calls_ffmpeg(tmp_path):
    input_mp4 = tmp_path / "BV1abc.mp4"
    input_mp4.write_bytes(b"fake")
    output_wav = tmp_path / "BV1abc.wav"
    with patch("speakerforge.stages.stage1_extract.ffmpeg") as mock_ff:
        mock_stream = MagicMock()
        mock_ff.input.return_value = mock_stream
        mock_stream.output.return_value = mock_stream
        mock_stream.overwrite_output.return_value = mock_stream
        extract_audio(input_mp4, output_wav, sample_rate=24000)
        mock_ff.input.assert_called_once_with(str(input_mp4))


def test_run_skips_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "raw_sources/bilibili/alice"
    raw.mkdir(parents=True)
    (raw / "BV1.mp4").write_bytes(b"fake")
    out = tmp_path / "processed/alice/audio"
    out.mkdir(parents=True)
    (out / "BV1.wav").write_bytes(b"exists")

    with patch("speakerforge.stages.stage1_extract.extract_audio") as mock_ex:
        run(PipelineConfig(stage1={"sample_rate": 24000}), speaker="alice")
        mock_ex.assert_not_called()
```

- [ ] **Step 3: Run tests, confirm they fail**

```bash
pytest tests/test_stage1.py -v
```

Expected: `ImportError: cannot import name 'extract_audio'`

- [ ] **Step 4: Write `speakerforge/stages/stage1_extract.py`**

```python
from __future__ import annotations
from pathlib import Path

import ffmpeg
from tqdm import tqdm

from speakerforge.config import PipelineConfig

RAW_DIR = Path("raw_sources/bilibili")
PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    sr = pipeline_cfg.stage1.get("sample_rate", 24000)
    raw_dir = RAW_DIR / speaker
    out_dir = PROC_DIR / speaker / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(raw_dir.glob("BV*.mp4"))
    if not mp4_files:
        raise FileNotFoundError(f"No MP4 files in {raw_dir}")

    for mp4 in tqdm(mp4_files, desc="Stage 1: extracting audio"):
        wav_out = out_dir / (mp4.stem + ".wav")
        if wav_out.exists():
            continue
        extract_audio(mp4, wav_out, sample_rate=sr)


def extract_audio(input_path: Path, output_path: Path, sample_rate: int = 24000) -> None:
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), ac=1, ar=sample_rate, vn=None, acodec="pcm_s16le")
        .overwrite_output()
        .run(quiet=True)
    )
```

- [ ] **Step 5: Run tests, confirm they pass**

```bash
pytest tests/test_stage1.py -v
```

Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add speakerforge/stages/stage1_extract.py tests/test_stage1.py tests/conftest.py
git commit -m "feat: stage1 audio extraction via ffmpeg"
```

---

## Task 7: Stage 2 — Demucs Vocal Isolation

**Files:**
- Create: `speakerforge/stages/stage2_demucs.py`

- [ ] **Step 1: Write `speakerforge/stages/stage2_demucs.py`**

```python
from __future__ import annotations
import shutil
from pathlib import Path

import soundfile as sf
import numpy as np
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    model_name = pipeline_cfg.stage2.get("model", "htdemucs_ft")
    in_dir = PROC_DIR / speaker / "audio"
    out_dir = PROC_DIR / speaker / "vocals"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(in_dir.glob("BV*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {in_dir}")

    model = _load_model(model_name)

    for wav in tqdm(wav_files, desc="Stage 2: vocal isolation"):
        out_wav = out_dir / wav.name
        if out_wav.exists():
            continue
        _separate(model, wav, out_wav)


def _load_model(model_name: str):
    from demucs.pretrained import get_model
    return get_model(model_name)


def _separate(model, input_wav: Path, output_wav: Path) -> None:
    import torch
    from demucs.apply import apply_model

    audio, sr = sf.read(str(input_wav), dtype="float32")
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]  # (1, T)
    else:
        audio = audio.T  # (C, T)

    # Demucs expects (batch, channels, time) as torch tensor
    wav_tensor = torch.from_numpy(audio).unsqueeze(0)  # (1, C, T)

    # Resample to model's expected sample rate if needed
    model_sr = model.samplerate
    if sr != model_sr:
        import torchaudio
        wav_tensor = torchaudio.functional.resample(wav_tensor, sr, model_sr)

    with torch.no_grad():
        sources = apply_model(model, wav_tensor, device="cpu")
    # sources shape: (1, n_sources, channels, time)
    # source order: drums, bass, other, vocals (for htdemucs_ft)
    stem_names = model.sources
    vocals_idx = stem_names.index("vocals")
    vocals = sources[0, vocals_idx]  # (channels, time)

    # Convert back to mono 24kHz
    vocals_mono = vocals.mean(dim=0).numpy()  # (time,)
    if model_sr != 24000:
        import torchaudio
        vocals_tensor = torch.from_numpy(vocals_mono).unsqueeze(0)
        vocals_tensor = torchaudio.functional.resample(vocals_tensor, model_sr, 24000)
        vocals_mono = vocals_tensor.squeeze(0).numpy()

    sf.write(str(output_wav), vocals_mono, 24000, subtype="PCM_16")
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from speakerforge.stages.stage2_demucs import run; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add speakerforge/stages/stage2_demucs.py
git commit -m "feat: stage2 Demucs htdemucs_ft vocal isolation"
```

---

## Task 8: Stage 3 — VAD Segmentation

**Files:**
- Create: `speakerforge/stages/stage3_vad.py`
- Create: `tests/test_stage3.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage3.py
import numpy as np
import pytest
import soundfile as sf
from speakerforge.stages.stage3_vad import merge_segments, filter_segments


def test_merge_close_segments():
    segments = [(0.0, 1.0), (1.2, 2.0), (3.0, 4.0)]
    merged = merge_segments(segments, min_silence=0.3)
    assert merged == [(0.0, 2.0), (3.0, 4.0)]


def test_merge_no_merge_when_gap_large():
    segments = [(0.0, 1.0), (2.0, 3.0)]
    merged = merge_segments(segments, min_silence=0.3)
    assert merged == [(0.0, 1.0), (2.0, 3.0)]


def test_filter_segments_by_duration():
    segments = [(0.0, 0.5), (1.0, 3.0), (5.0, 14.0)]
    filtered = filter_segments(segments, min_dur=0.8, max_dur=8.0)
    assert filtered == [(1.0, 3.0)]


def test_filter_empty():
    assert filter_segments([], min_dur=0.8, max_dur=8.0) == []
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest tests/test_stage3.py -v
```

Expected: `ImportError: cannot import name 'merge_segments'`

- [ ] **Step 3: Write `speakerforge/stages/stage3_vad.py`**

```python
from __future__ import annotations
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    cfg = pipeline_cfg.stage3
    min_dur = cfg.get("min_duration", 0.8)
    max_dur = cfg.get("max_duration", 8.0)
    min_silence = cfg.get("min_silence_duration", 0.3)

    in_dir = PROC_DIR / speaker / "vocals"
    out_dir = PROC_DIR / speaker / "segments"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(in_dir.glob("BV*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No WAV files in {in_dir}")

    vad_model, utils = _load_silero_vad()
    (get_speech_timestamps, _, read_audio, *_) = utils

    for wav in tqdm(wav_files, desc="Stage 3: VAD segmentation"):
        _segment_file(wav, out_dir, vad_model, get_speech_timestamps, read_audio,
                      min_dur, max_dur, min_silence)


def _load_silero_vad():
    import torch
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    return model, utils


def _segment_file(wav_path, out_dir, model, get_speech_timestamps, read_audio,
                  min_dur, max_dur, min_silence):
    import torch
    audio = read_audio(str(wav_path), sampling_rate=16000)
    timestamps = get_speech_timestamps(audio, model, sampling_rate=16000,
                                       return_seconds=True)
    raw_segments = [(t["start"], t["end"]) for t in timestamps]
    segments = filter_segments(merge_segments(raw_segments, min_silence), min_dur, max_dur)

    full_audio, sr = sf.read(str(wav_path), dtype="float32")

    for i, (start, end) in enumerate(segments):
        s = int(start * sr)
        e = int(end * sr)
        chunk = full_audio[s:e]
        out_name = f"{wav_path.stem}_{i:04d}.wav"
        sf.write(str(out_dir / out_name), chunk, sr, subtype="PCM_16")


def merge_segments(segments: list[tuple[float, float]], min_silence: float) -> list[tuple[float, float]]:
    if not segments:
        return []
    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_end = merged[-1][1]
        if start - prev_end < min_silence:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def filter_segments(segments: list[tuple[float, float]],
                    min_dur: float, max_dur: float) -> list[tuple[float, float]]:
    return [(s, e) for s, e in segments if min_dur <= (e - s) <= max_dur]
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
pytest tests/test_stage3.py -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add speakerforge/stages/stage3_vad.py tests/test_stage3.py
git commit -m "feat: stage3 silero-vad segmentation with merge and filter logic"
```

---

## Task 9: Stage 4 — WhisperX Forced Alignment

**Files:**
- Create: `speakerforge/stages/stage4_align.py`

- [ ] **Step 1: Write `speakerforge/stages/stage4_align.py`**

```python
from __future__ import annotations
import json
from pathlib import Path

from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    cfg = pipeline_cfg.stage4
    model_size = cfg.get("model", "large-v3")
    language = cfg.get("language", "zh")
    batch_size = cfg.get("batch_size", 16)

    in_dir = PROC_DIR / speaker / "segments"
    out_dir = PROC_DIR / speaker / "aligned"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(in_dir.glob("*.wav"))
    if not wav_files:
        raise FileNotFoundError(f"No segment WAVs in {in_dir}")

    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading WhisperX model: {model_size} on {device}")
    model = whisperx.load_model(model_size, device, compute_type=compute_type, language=language)
    align_model, metadata = whisperx.load_align_model(language_code=language, device=device)

    for wav in tqdm(wav_files, desc="Stage 4: WhisperX alignment"):
        json_out = out_dir / (wav.stem + ".json")
        if json_out.exists():
            continue
        _align_file(wav, json_out, model, align_model, metadata, device, batch_size)


def _align_file(wav_path, json_out, model, align_model, metadata, device, batch_size):
    import whisperx

    audio = whisperx.load_audio(str(wav_path))
    result = model.transcribe(audio, batch_size=batch_size)

    if not result.get("segments"):
        json_out.write_text(json.dumps({"text": "", "segments": [], "confidence": 0.0},
                                       ensure_ascii=False), encoding="utf-8")
        return

    aligned = whisperx.align(result["segments"], align_model, metadata, audio, device,
                              return_char_alignments=False)

    text = " ".join(s["text"] for s in aligned["segments"]).strip()
    confidence = _mean_confidence(aligned["segments"])

    json_out.write_text(json.dumps({
        "text": text,
        "segments": aligned["segments"],
        "confidence": confidence,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _mean_confidence(segments: list[dict]) -> float:
    scores = [w.get("score", 0.0) for seg in segments for w in seg.get("words", [])]
    return float(sum(scores) / len(scores)) if scores else 0.0
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from speakerforge.stages.stage4_align import run; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add speakerforge/stages/stage4_align.py
git commit -m "feat: stage4 WhisperX forced alignment with per-segment JSON output"
```

---

## Task 10: Stage 5 — Quality Filtering

**Files:**
- Create: `speakerforge/stages/stage5_filter.py`
- Create: `tests/test_stage5.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage5.py
import json
import pytest
from pathlib import Path
from speakerforge.stages.stage5_filter import passes_filter, FilterConfig


def test_passes_filter_good_segment():
    entry = {"text": "这是一个测试句子", "duration": 3.0, "confidence": 0.85}
    cfg = FilterConfig(min_duration=1.0, max_duration=6.0, min_confidence=0.6, min_text_chars=4)
    assert passes_filter(entry, cfg) is True


def test_rejects_too_short():
    entry = {"text": "好的", "duration": 0.5, "confidence": 0.9}
    cfg = FilterConfig(min_duration=1.0, max_duration=6.0, min_confidence=0.6, min_text_chars=4)
    assert passes_filter(entry, cfg) is False


def test_rejects_too_long():
    entry = {"text": "这是很长的句子" * 10, "duration": 9.0, "confidence": 0.9}
    cfg = FilterConfig(min_duration=1.0, max_duration=6.0, min_confidence=0.6, min_text_chars=4)
    assert passes_filter(entry, cfg) is False


def test_rejects_low_confidence():
    entry = {"text": "这是测试", "duration": 2.0, "confidence": 0.4}
    cfg = FilterConfig(min_duration=1.0, max_duration=6.0, min_confidence=0.6, min_text_chars=4)
    assert passes_filter(entry, cfg) is False


def test_rejects_short_text():
    entry = {"text": "嗯", "duration": 2.0, "confidence": 0.9}
    cfg = FilterConfig(min_duration=1.0, max_duration=6.0, min_confidence=0.6, min_text_chars=4)
    assert passes_filter(entry, cfg) is False
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest tests/test_stage5.py -v
```

Expected: `ImportError: cannot import name 'passes_filter'`

- [ ] **Step 3: Write `speakerforge/stages/stage5_filter.py`**

```python
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


@dataclass
class FilterConfig:
    min_duration: float = 1.0
    max_duration: float = 6.0
    min_confidence: float = 0.6
    min_text_chars: int = 4


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    cfg_dict = pipeline_cfg.stage5
    cfg = FilterConfig(
        min_duration=cfg_dict.get("min_duration", 1.0),
        max_duration=cfg_dict.get("max_duration", 6.0),
        min_confidence=cfg_dict.get("min_confidence", 0.6),
        min_text_chars=cfg_dict.get("min_text_chars", 4),
    )

    seg_dir = PROC_DIR / speaker / "segments"
    align_dir = PROC_DIR / speaker / "aligned"
    out_dir = PROC_DIR / speaker / "filtered"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    json_files = sorted(align_dir.glob("*.json"))

    for jf in tqdm(json_files, desc="Stage 5: filtering"):
        data = json.loads(jf.read_text(encoding="utf-8"))
        wav_path = seg_dir / (jf.stem + ".wav")
        if not wav_path.exists():
            continue
        info = sf.info(str(wav_path))
        duration = info.duration
        entry = {"text": data.get("text", ""), "duration": duration,
                 "confidence": data.get("confidence", 0.0),
                 "segment_path": str(wav_path)}
        if passes_filter(entry, cfg):
            manifest.append(entry)

    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for e in manifest:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    total_dur = sum(e["duration"] for e in manifest)
    print(f"Kept {len(manifest)} segments, {total_dur/60:.1f} min")


def passes_filter(entry: dict, cfg: FilterConfig) -> bool:
    if entry["duration"] < cfg.min_duration or entry["duration"] > cfg.max_duration:
        return False
    if entry["confidence"] < cfg.min_confidence:
        return False
    if len(entry["text"].strip()) < cfg.min_text_chars:
        return False
    return True
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
pytest tests/test_stage5.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add speakerforge/stages/stage5_filter.py tests/test_stage5.py
git commit -m "feat: stage5 quality filtering with manifest.jsonl output"
```

---

## Task 11: Stage 6 — Normalization

**Files:**
- Create: `speakerforge/stages/stage6_normalize.py`

- [ ] **Step 1: Write `speakerforge/stages/stage6_normalize.py`**

```python
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")


def run(pipeline_cfg: PipelineConfig, speaker: str) -> None:
    cfg = pipeline_cfg.stage6
    target_rms_dbfs = cfg.get("target_rms_dbfs", -20)
    silence_threshold_dbfs = cfg.get("silence_threshold_dbfs", -40)

    filtered_dir = PROC_DIR / speaker / "filtered"
    out_dir = PROC_DIR / speaker / "normalized"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = filtered_dir / "manifest.jsonl"
    entries = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    for entry in tqdm(entries, desc="Stage 6: normalizing"):
        src = Path(entry["segment_path"])
        dst = out_dir / src.name
        if dst.exists():
            continue
        audio, sr = sf.read(str(src), dtype="float32")
        audio = _trim_silence(audio, sr, silence_threshold_dbfs)
        audio = _rms_normalize(audio, target_rms_dbfs)
        sf.write(str(dst), audio, sr, subtype="PCM_16")


def _rms_normalize(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_dbfs / 20)
    return audio * (target_rms / rms)


def _trim_silence(audio: np.ndarray, sr: int, threshold_dbfs: float,
                  min_silence_sec: float = 0.1) -> np.ndarray:
    threshold_linear = 10 ** (threshold_dbfs / 20)
    min_samples = int(min_silence_sec * sr)
    above = np.abs(audio) > threshold_linear

    # Find first and last non-silent sample
    nonzero = np.where(above)[0]
    if len(nonzero) == 0:
        return audio
    start = max(0, nonzero[0] - min_samples)
    end = min(len(audio), nonzero[-1] + min_samples)
    return audio[start:end]
```

- [ ] **Step 2: Verify import works**

```bash
python -c "from speakerforge.stages.stage6_normalize import run; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add speakerforge/stages/stage6_normalize.py
git commit -m "feat: stage6 RMS normalization and silence trimming"
```

---

## Task 12: Stage 7 — Dataset Packaging

**Files:**
- Create: `speakerforge/stages/stage7_package.py`
- Create: `tests/test_stage7.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage7.py
import json
import shutil
import pandas as pd
import pytest
import soundfile as sf
import numpy as np
from pathlib import Path
from speakerforge.stages.stage7_package import package_dataset
from speakerforge.config import PipelineConfig


def test_package_dataset(tmp_path, make_wav):
    normalized_dir = tmp_path / "processed/alice/normalized"
    normalized_dir.mkdir(parents=True)

    # Create 3 fake normalized WAV files
    for i in range(3):
        wav = make_wav(f"processed/alice/normalized/BV1_{i:04d}.wav", duration=2.0)

    # Create manifest
    manifest_path = tmp_path / "processed/alice/filtered/manifest.jsonl"
    manifest_path.parent.mkdir(parents=True)
    entries = [
        {"text": f"句子{i}", "duration": 2.0, "confidence": 0.9,
         "segment_path": str(normalized_dir / f"BV1_{i:04d}.wav")}
        for i in range(3)
    ]
    with manifest_path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    out_dir = tmp_path / "dataset/alice_versionB"
    package_dataset(
        normalized_dir=normalized_dir,
        manifest_path=manifest_path,
        out_dir=out_dir,
    )

    csv = out_dir / "metadata.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert len(df) == 3
    assert "file_path" in df.columns
    assert "text" in df.columns
    assert "duration" in df.columns
    assert (out_dir / "audio").is_dir()
    assert len(list((out_dir / "audio").glob("*.wav"))) == 3
```

- [ ] **Step 2: Run test, confirm it fails**

```bash
pytest tests/test_stage7.py -v
```

Expected: `ImportError: cannot import name 'package_dataset'`

- [ ] **Step 3: Write `speakerforge/stages/stage7_package.py`**

```python
from __future__ import annotations
import json
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from speakerforge.config import PipelineConfig

PROC_DIR = Path("processed")
DATASET_DIR = Path("dataset")


def run(pipeline_cfg: PipelineConfig, speaker: str, version: str = "B") -> None:
    normalized_dir = PROC_DIR / speaker / "normalized"
    manifest_path = PROC_DIR / speaker / "filtered" / "manifest.jsonl"
    out_dir = DATASET_DIR / f"{speaker}_version{version}"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Run stage 5 first: {manifest_path}")

    package_dataset(normalized_dir, manifest_path, out_dir)
    print(f"Dataset written to {out_dir}")


def package_dataset(normalized_dir: Path, manifest_path: Path, out_dir: Path) -> None:
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    entries = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    rows = []
    for i, entry in enumerate(tqdm(entries, desc="Stage 7: packaging")):
        src = Path(entry["segment_path"])
        # Use normalized file if available, fall back to segment path
        norm_path = normalized_dir / src.name
        src = norm_path if norm_path.exists() else src

        dst_name = f"{i+1:04d}.wav"
        dst = audio_dir / dst_name
        shutil.copy2(str(src), str(dst))

        rows.append({
            "file_path": f"audio/{dst_name}",
            "text": entry["text"],
            "duration": round(entry["duration"], 3),
            "confidence": round(entry.get("confidence", 0.0), 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metadata.csv", index=False, encoding="utf-8")

    total_min = df["duration"].sum() / 60
    print(f"  {len(df)} segments | {total_min:.1f} min total")
    print(f"  Duration: mean={df['duration'].mean():.1f}s, "
          f"min={df['duration'].min():.1f}s, max={df['duration'].max():.1f}s")
```

- [ ] **Step 4: Run test, confirm it passes**

```bash
pytest tests/test_stage7.py -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add speakerforge/stages/stage7_package.py tests/test_stage7.py
git commit -m "feat: stage7 LJSpeech-format dataset packaging"
```

---

## Task 13: Run Full Test Suite

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass. Typical output:
```
tests/test_bili_auth.py::test_load_credential_returns_none_if_missing PASSED
tests/test_bili_auth.py::test_load_credential_returns_credential PASSED
tests/test_bili_auth.py::test_save_credential_writes_json PASSED
tests/test_bili_auth.py::test_get_or_login_returns_existing PASSED
tests/test_config.py::test_load_sources_video PASSED
tests/test_config.py::test_load_sources_user_videos PASSED
tests/test_config.py::test_load_pipeline_defaults PASSED
tests/test_config.py::test_bilibili_source_optional_fields PASSED
tests/test_stage1.py::test_extract_audio_calls_ffmpeg PASSED
tests/test_stage1.py::test_run_skips_existing PASSED
tests/test_stage3.py::test_merge_close_segments PASSED
tests/test_stage3.py::test_merge_no_merge_when_gap_large PASSED
tests/test_stage3.py::test_filter_segments_by_duration PASSED
tests/test_stage3.py::test_filter_empty PASSED
tests/test_stage5.py::test_passes_filter_good_segment PASSED
tests/test_stage5.py::test_rejects_too_short PASSED
tests/test_stage5.py::test_rejects_too_long PASSED
tests/test_stage5.py::test_rejects_low_confidence PASSED
tests/test_stage5.py::test_rejects_short_text PASSED
tests/test_stage7.py::test_package_dataset PASSED
```

- [ ] **Step 2: Verify CLI shows all commands**

```bash
python -m speakerforge --help
```

Expected: lists `login`, `run`, `stage0` through `stage7`, `stats`

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: complete SpeakerForge dataset pipeline stages 0-7 with tests"
```

---

## Self-Review

**Spec coverage:**
- [x] Stage 0: Config-driven whitelist, `video`/`collection`/`user_videos` types, `limit`, idempotent download
- [x] Stage 1: ffmpeg → mono 24kHz WAV
- [x] Stage 2: Demucs `htdemucs_ft` vocal isolation
- [x] Stage 3: silero-vad, merge close segments, filter 0.8s–8s
- [x] Stage 4: WhisperX `large-v3`, `language=zh`, word-level alignment, confidence score
- [x] Stage 5: Filter by duration/confidence/text length, `manifest.jsonl` output
- [x] Stage 6: RMS normalize to -20 dBFS, trim silence
- [x] Stage 7: `audio/<N>.wav` + `metadata.csv`, LJSpeech format, version A/B naming
- [x] `--dry-run` on stage0
- [x] `run --stages 0-7` for chaining
- [x] `stats` command
- [x] Idempotency: each stage skips already-processed files
- [x] Cookie file configurable via `pipeline.yaml`

**Type consistency check:**
- `load_pipeline` returns `PipelineConfig` — used correctly in all stages
- `passes_filter(entry, cfg)` — `FilterConfig` dataclass matches usage in tests
- `package_dataset(normalized_dir, manifest_path, out_dir)` — matches test call
- `merge_segments(segments, min_silence)` / `filter_segments(segments, min_dur, max_dur)` — match test calls

**No placeholders found.**
