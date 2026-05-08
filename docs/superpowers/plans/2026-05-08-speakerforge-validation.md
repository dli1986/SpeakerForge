# SpeakerForge Validation Pipeline (Stages 8–9) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate dataset quality by fine-tuning a CosyVoice 2 LoRA adapter on the packaged dataset, then running inference and scoring speaker similarity to compare Version A (baseline) vs Version B (full pipeline) datasets.

**Architecture:** Stage 8 wraps the official CosyVoice 2 fine-tuning script (`finetune.py`) from a separately-cloned repo, converting the `metadata.csv` into its expected data format, then launching training via subprocess. Stage 9 runs CosyVoice 2 inference with the trained adapter and scores speaker similarity using resemblyzer cosine distance.

**Prerequisites:**
- Stage 7 must have produced `dataset/<speaker>_versionA/` and `dataset/<speaker>_versionB/`
- CosyVoice 2 repo must be cloned: `git clone https://github.com/FunAudioLLM/CosyVoice`
- `cosyvoice2_repo` path set correctly in `pipeline.yaml` stage8 section
- GPU with ≥16GB VRAM recommended for fine-tuning

**Tech Stack:** CosyVoice 2 (official scripts), resemblyzer, soundfile, pandas, torch, click

---

## File Map

| File | Responsibility |
|---|---|
| `speakerforge/stages/stage8_finetune.py` | Convert dataset → CosyVoice format; launch fine-tuning subprocess |
| `speakerforge/stages/stage9_eval.py` | Inference with LoRA adapter; resemblyzer speaker similarity scoring |
| `tests/test_stage8.py` | Dataset conversion logic tests (no GPU needed) |
| `tests/test_stage9.py` | Score aggregation and output format tests |

---

## Task 1: Understand CosyVoice 2 Fine-tuning Interface

- [ ] **Step 1: Clone CosyVoice 2 and inspect fine-tuning scripts**

```bash
git clone https://github.com/FunAudioLLM/CosyVoice /path/to/CosyVoice
cd /path/to/CosyVoice
ls examples/  # look for finetune or lora examples
```

- [ ] **Step 2: Check what data format CosyVoice 2 finetune expects**

```bash
# Typically CosyVoice expects a data.list file with lines:
# {"key": "0001", "wav": "/abs/path/0001.wav", "text": "...", "spk": "speaker_name"}
# Confirm by reading:
cat examples/libritts/cosyvoice2/run.sh
# or
cat tools/make_cosyvoice_fbank.py
```

Expected: Find the data format (usually a JSONL `.list` file with `key`, `wav`, `text`, `spk` fields).

- [ ] **Step 3: Update `pipeline.yaml` with CosyVoice 2 repo path**

```yaml
stage8:
  base_model: cosyvoice2
  cosyvoice2_repo: "/path/to/CosyVoice"   # <-- set your actual path here
  lora_rank: 16
  epochs: 10
  batch_size: 4
```

---

## Task 2: Stage 8 — CosyVoice 2 LoRA Fine-tuning

**Files:**
- Create: `speakerforge/stages/stage8_finetune.py`
- Create: `tests/test_stage8.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage8.py
import json
import pandas as pd
import pytest
from pathlib import Path
from speakerforge.stages.stage8_finetune import convert_to_cosyvoice_format


def test_convert_creates_data_list(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    for i in range(3):
        (audio_dir / f"{i+1:04d}.wav").write_bytes(b"fake")

    csv = tmp_path / "metadata.csv"
    pd.DataFrame([
        {"file_path": f"audio/{i+1:04d}.wav", "text": f"句子{i}", "duration": 2.0, "confidence": 0.9}
        for i in range(3)
    ]).to_csv(csv, index=False)

    out_list = tmp_path / "data.list"
    convert_to_cosyvoice_format(
        dataset_dir=tmp_path,
        speaker="alice",
        out_list=out_list,
    )

    lines = [json.loads(l) for l in out_list.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    assert lines[0]["spk"] == "alice"
    assert lines[0]["text"] == "句子0"
    assert lines[0]["key"] == "0001"
    assert Path(lines[0]["wav"]).is_absolute()


def test_convert_skips_empty_text(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    (audio_dir / "0001.wav").write_bytes(b"fake")

    csv = tmp_path / "metadata.csv"
    pd.DataFrame([
        {"file_path": "audio/0001.wav", "text": "", "duration": 2.0, "confidence": 0.9},
    ]).to_csv(csv, index=False)

    out_list = tmp_path / "data.list"
    convert_to_cosyvoice_format(tmp_path, "alice", out_list)
    lines = [l for l in out_list.read_text().splitlines() if l.strip()]
    assert len(lines) == 0
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
cd c:/Users/dli/Projects/MyTest/SpeakerForge
pytest tests/test_stage8.py -v
```

Expected: `ImportError: cannot import name 'convert_to_cosyvoice_format'`

- [ ] **Step 3: Write `speakerforge/stages/stage8_finetune.py`**

```python
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from speakerforge.config import PipelineConfig

DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")


def run(pipeline_cfg: PipelineConfig, speaker: str, version: str = "B") -> None:
    cfg = pipeline_cfg.stage8
    cosyvoice_repo = Path(cfg.get("cosyvoice2_repo", "")).expanduser()
    if not cosyvoice_repo.exists():
        raise FileNotFoundError(
            f"CosyVoice 2 repo not found at: {cosyvoice_repo}\n"
            "Clone it: git clone https://github.com/FunAudioLLM/CosyVoice\n"
            "Then set cosyvoice2_repo in pipeline.yaml"
        )

    dataset_dir = DATASET_DIR / f"{speaker}_version{version}"
    if not (dataset_dir / "metadata.csv").exists():
        raise FileNotFoundError(f"Run stage7 first: {dataset_dir / 'metadata.csv'}")

    model_dir = MODELS_DIR / f"{speaker}_version{version}" / "cosyvoice2_lora"
    model_dir.mkdir(parents=True, exist_ok=True)

    data_list = model_dir / "data.list"
    convert_to_cosyvoice_format(dataset_dir, speaker, data_list)

    _run_finetune(
        cosyvoice_repo=cosyvoice_repo,
        data_list=data_list,
        model_dir=model_dir,
        speaker=speaker,
        lora_rank=cfg.get("lora_rank", 16),
        epochs=cfg.get("epochs", 10),
        batch_size=cfg.get("batch_size", 4),
    )


def convert_to_cosyvoice_format(dataset_dir: Path, speaker: str, out_list: Path) -> None:
    df = pd.read_csv(dataset_dir / "metadata.csv")
    lines = []
    for _, row in df.iterrows():
        text = str(row["text"]).strip()
        if not text:
            continue
        wav_path = (dataset_dir / row["file_path"]).resolve()
        key = Path(row["file_path"]).stem
        lines.append(json.dumps({
            "key": key,
            "wav": str(wav_path),
            "text": text,
            "spk": speaker,
        }, ensure_ascii=False))
    out_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} entries to {out_list}")


def _run_finetune(cosyvoice_repo: Path, data_list: Path, model_dir: Path,
                  speaker: str, lora_rank: int, epochs: int, batch_size: int) -> None:
    # CosyVoice 2 fine-tuning entry point — adjust script path if repo structure differs
    finetune_script = cosyvoice_repo / "examples" / "finetune" / "finetune.py"
    if not finetune_script.exists():
        # Try alternate location
        finetune_script = cosyvoice_repo / "finetune.py"
    if not finetune_script.exists():
        raise FileNotFoundError(
            f"CosyVoice 2 finetune script not found. "
            f"Check repo structure at {cosyvoice_repo}"
        )

    cmd = [
        sys.executable,
        str(finetune_script),
        "--train_data", str(data_list),
        "--output_dir", str(model_dir),
        "--speaker", speaker,
        "--lora_rank", str(lora_rank),
        "--num_epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lora_only",  # freeze base weights
    ]
    log_path = model_dir / "train_log.txt"
    print(f"Running CosyVoice 2 fine-tuning. Log: {log_path}")
    with log_path.open("w") as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(cosyvoice_repo))
    if result.returncode != 0:
        raise RuntimeError(
            f"Fine-tuning failed (exit {result.returncode}). "
            f"See {log_path} for details."
        )
    print(f"Fine-tuning complete. Model saved to {model_dir}")
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
pytest tests/test_stage8.py -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add speakerforge/stages/stage8_finetune.py tests/test_stage8.py
git commit -m "feat: stage8 CosyVoice 2 LoRA fine-tuning wrapper"
```

---

## Task 3: Stage 9 — Inference + Evaluation

**Files:**
- Create: `speakerforge/stages/stage9_eval.py`
- Create: `tests/test_stage9.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_stage9.py
import json
import numpy as np
import pytest
from speakerforge.stages.stage9_eval import aggregate_scores, ScoreResult


def test_aggregate_scores_mean():
    scores = [0.8, 0.75, 0.85, 0.9]
    result = aggregate_scores(scores)
    assert abs(result.mean - 0.825) < 0.001
    assert result.min == 0.75
    assert result.max == 0.9
    assert result.passes_threshold(0.75) is True


def test_aggregate_scores_fails_threshold():
    scores = [0.6, 0.65, 0.7]
    result = aggregate_scores(scores)
    assert result.passes_threshold(0.75) is False


def test_aggregate_empty_scores():
    result = aggregate_scores([])
    assert result.mean == 0.0
```

- [ ] **Step 2: Run tests, confirm they fail**

```bash
pytest tests/test_stage9.py -v
```

Expected: `ImportError: cannot import name 'aggregate_scores'`

- [ ] **Step 3: Write `speakerforge/stages/stage9_eval.py`**

```python
from __future__ import annotations
import json
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from speakerforge.config import PipelineConfig

MODELS_DIR = Path("models")
EVAL_DIR = Path("eval")
DATASET_DIR = Path("dataset")

TEST_PROMPTS = [
    "你好，今天天气怎么样？",
    "我觉得这个项目非常有意思，值得深入研究。",
    "深度学习在语音合成领域取得了显著的进展。",
    "请帮我预订一张明天下午三点从北京到上海的高铁票。",
    "人工智能技术正在深刻改变我们的生产方式和生活方式，带来了前所未有的机遇与挑战。",
]


@dataclass
class ScoreResult:
    mean: float
    min: float
    max: float
    scores: list[float]

    def passes_threshold(self, threshold: float) -> bool:
        return self.mean >= threshold


def run(pipeline_cfg: PipelineConfig, speaker: str, version: str = "B") -> None:
    cosyvoice_repo = Path(pipeline_cfg.stage8.get("cosyvoice2_repo", "")).expanduser()
    model_dir = MODELS_DIR / f"{speaker}_version{version}" / "cosyvoice2_lora"
    out_dir = EVAL_DIR / f"{speaker}_version{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_out = out_dir / "audio"
    audio_out.mkdir(exist_ok=True)

    if not model_dir.exists():
        raise FileNotFoundError(f"Run stage8 first: {model_dir}")

    # Get reference clips for resemblyzer (first 3 clips from dataset)
    dataset_dir = DATASET_DIR / f"{speaker}_version{version}" / "audio"
    ref_wavs = sorted(dataset_dir.glob("*.wav"))[:3]
    if not ref_wavs:
        raise FileNotFoundError(f"No reference WAVs in {dataset_dir}")

    reference_embedding = _get_reference_embedding(ref_wavs)

    generated_wavs = []
    for i, prompt in enumerate(TEST_PROMPTS):
        out_wav = audio_out / f"prompt_{i+1:02d}.wav"
        if not out_wav.exists():
            _run_inference(cosyvoice_repo, model_dir, prompt, out_wav, speaker)
        generated_wavs.append(out_wav)

    scores = [_cosine_similarity(reference_embedding, _embed_wav(w))
              for w in generated_wavs if w.exists()]

    result = aggregate_scores(scores)
    scores_data = {
        "speaker": speaker,
        "version": version,
        "similarity_threshold": 0.75,
        "passes": result.passes_threshold(0.75),
        "mean_similarity": round(result.mean, 4),
        "min_similarity": round(result.min, 4),
        "max_similarity": round(result.max, 4),
        "per_prompt": [
            {"prompt": p, "score": round(s, 4)}
            for p, s in zip(TEST_PROMPTS, result.scores)
        ],
    }
    scores_path = out_dir / "scores.json"
    scores_path.write_text(json.dumps(scores_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nVersion {version} evaluation:")
    print(f"  Mean speaker similarity: {result.mean:.3f} "
          f"({'PASS' if result.passes_threshold(0.75) else 'FAIL'} @ 0.75 threshold)")
    print(f"  Range: [{result.min:.3f}, {result.max:.3f}]")
    print(f"  Scores saved to {scores_path}")


def _get_reference_embedding(ref_wavs: list[Path]) -> np.ndarray:
    embeddings = [_embed_wav(w) for w in ref_wavs]
    return np.mean(embeddings, axis=0)


def _embed_wav(wav_path: Path) -> np.ndarray:
    from resemblyzer import VoiceEncoder, preprocess_wav
    encoder = VoiceEncoder()
    wav = preprocess_wav(str(wav_path))
    return encoder.embed_utterance(wav)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


def _run_inference(cosyvoice_repo: Path, model_dir: Path, text: str,
                   out_wav: Path, speaker: str) -> None:
    infer_script = cosyvoice_repo / "inference.py"
    if not infer_script.exists():
        infer_script = cosyvoice_repo / "examples" / "inference" / "inference.py"
    if not infer_script.exists():
        raise FileNotFoundError(
            f"CosyVoice 2 inference script not found at {cosyvoice_repo}. "
            "Check repo structure."
        )
    cmd = [
        sys.executable, str(infer_script),
        "--model_dir", str(model_dir),
        "--text", text,
        "--spk", speaker,
        "--output_wav", str(out_wav),
    ]
    result = subprocess.run(cmd, cwd=str(cosyvoice_repo), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Inference failed for: {text}\n{result.stderr}")


def aggregate_scores(scores: list[float]) -> ScoreResult:
    if not scores:
        return ScoreResult(mean=0.0, min=0.0, max=0.0, scores=[])
    arr = np.array(scores, dtype=float)
    return ScoreResult(
        mean=float(arr.mean()),
        min=float(arr.min()),
        max=float(arr.max()),
        scores=list(scores),
    )
```

- [ ] **Step 4: Run tests, confirm they pass**

```bash
pytest tests/test_stage9.py -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add speakerforge/stages/stage9_eval.py tests/test_stage9.py
git commit -m "feat: stage9 CosyVoice 2 inference + resemblyzer speaker similarity scoring"
```

---

## Task 4: Add stage8/stage9 CLI Commands

**Files:**
- Modify: `speakerforge/cli.py`

- [ ] **Step 1: Add stage8 and stage9 commands to `speakerforge/cli.py`**

Append after the existing `stage7` command:

```python
@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--version", default="B", type=click.Choice(["A", "B"]))
def stage8(speaker, config, version):
    """Stage 8: CosyVoice 2 LoRA fine-tuning."""
    from speakerforge.stages.stage8_finetune import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker, version=version)


@cli.command()
@click.option("--speaker", required=True)
@click.option("--config", default=DEFAULT_PIPELINE, show_default=True)
@click.option("--version", default="B", type=click.Choice(["A", "B"]))
def stage9(speaker, config, version):
    """Stage 9: Inference + speaker similarity evaluation."""
    from speakerforge.stages.stage9_eval import run
    from speakerforge.config import load_pipeline
    run(load_pipeline(config), speaker=speaker, version=version)
```

Also update the `stage_fns` dict in the `run` command to include stages 8 and 9:

```python
        8: lambda: __import__("speakerforge.stages.stage8_finetune", fromlist=["run"]).run(pipe_cfg, speaker=speaker, version=version),
        9: lambda: __import__("speakerforge.stages.stage9_eval", fromlist=["run"]).run(pipe_cfg, speaker=speaker, version=version),
```

- [ ] **Step 2: Verify CLI lists stage8 and stage9**

```bash
python -m speakerforge --help
```

Expected: output includes `stage8` and `stage9`

- [ ] **Step 3: Commit**

```bash
git add speakerforge/cli.py
git commit -m "feat: add stage8 and stage9 CLI subcommands"
```

---

## Task 5: Experiment Comparison Workflow

- [ ] **Step 1: Build Version A dataset (baseline — no Demucs/VAD/WhisperX)**

Run stages 0–1, then a simplified version of stages 4–7 that skips Demucs and VAD:

```bash
# Stage 0: download
python -m speakerforge stage0 --speaker <your_speaker>

# Stage 1: extract audio
python -m speakerforge stage1 --speaker <your_speaker>

# For Version A: copy stage1 output directly to segments/ (skip Demucs+VAD)
# Then run WhisperX directly on full audio (no alignment, just raw segments)
# Then package as version A:
python -m speakerforge stage7 --speaker <your_speaker> --version A
```

Note: Version A skips stages 2–6. To properly build a "naive" baseline, manually copy `processed/<speaker>/audio/*.wav` → `processed/<speaker>/segments/` and create a minimal `manifest.jsonl` using whisper (not whisperX) output for Version A.

- [ ] **Step 2: Build Version B dataset (full pipeline)**

```bash
python -m speakerforge run --stages 0-7 --speaker <your_speaker> --version B
```

- [ ] **Step 3: Fine-tune and evaluate both versions**

```bash
python -m speakerforge stage8 --speaker <your_speaker> --version A
python -m speakerforge stage8 --speaker <your_speaker> --version B

python -m speakerforge stage9 --speaker <your_speaker> --version A
python -m speakerforge stage9 --speaker <your_speaker> --version B
```

- [ ] **Step 4: Compare results**

```bash
python -m speakerforge stats --speaker <your_speaker>
cat eval/<your_speaker>_versionA/scores.json
cat eval/<your_speaker>_versionB/scores.json
```

Listen to `eval/<your_speaker>_versionA/audio/` vs `eval/<your_speaker>_versionB/audio/` for qualitative comparison.

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete SpeakerForge validation pipeline stages 8-9"
```

---

## Self-Review

**Spec coverage:**
- [x] Stage 8: LoRA only, base weights frozen, configurable rank/epochs/batch_size
- [x] Stage 8: No full fine-tuning enforced (`--lora_only` flag)
- [x] Stage 8: CosyVoice 2 as base model
- [x] Stage 9: Speaker similarity scoring (resemblyzer cosine)
- [x] Stage 9: Short + long + unseen prompts (5 fixed prompts)
- [x] Stage 9: `scores.json` output with per-prompt breakdown
- [x] Experiment comparison: Version A vs B both supported
- [x] `eval/<speaker>_versionA/` vs `eval/<speaker>_versionB/` dir naming

**Note on CosyVoice 2 script flags:** The `--lora_only`, `--lora_rank`, and CLI argument names in `_run_finetune()` and `_run_inference()` are based on the expected CosyVoice 2 interface. **Verify these against the actual cloned repo's `finetune.py` CLI and adjust argument names in `stage8_finetune.py:_run_finetune()` and `stage9_eval.py:_run_inference()` before running.** The data conversion logic (`convert_to_cosyvoice_format`) is independently tested and correct.

**Type consistency:** `ScoreResult.passes_threshold(threshold)` used in tests matches definition. `aggregate_scores(scores) -> ScoreResult` matches both tests and `run()` call.
