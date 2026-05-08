# SpeakerForge — Design Document

## Overview

SpeakerForge is a modular, config-driven pipeline that converts explicitly-listed Bilibili videos into high-quality single-speaker LoRA-ready speech datasets, then validates dataset quality via CosyVoice 2 fine-tuning and inference.

**Goals:**
- Quality over scale: 30–60 min of clean audio beats hours of noisy data
- Each stage independently runnable with cached intermediate outputs
- Validation is real: actual LoRA fine-tuning + listening test, not just metrics

**Non-goals:** UI, real-time processing, multi-speaker datasets, YouTube support, auto-crawling.

---

## Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Bilibili source resolution | `bilibili_api` (resolve collection/user_videos → BV list) | Supports `limit`, collection expansion, user video listing |
| Bilibili download | `yt-dlp` with Bilibili extractor + cookie file | Battle-tested codec handling, rate limiting, retries |
| Auth reuse | Copy ~50-line credential loader from `Subtitle/uploader/bilibili_uploader.py` | Self-contained; no fragile cross-project imports |
| Cookie sharing | Point to `../Subtitle/cache/bili_cookies.json` (configurable) | Reuse existing logged-in session, no re-scan |
| VAD | silero-vad | No HuggingFace token needed; runs on CPU; fast |
| Demucs model | `htdemucs_ft` | Highest vocal quality among Demucs models |
| ASR + alignment | WhisperX `large-v3`, `language=zh` | Forced alignment required; raw Whisper timestamps unacceptable |
| LoRA base model | CosyVoice 2 (official fine-tuning scripts) | Strong Chinese TTS, established LoRA workflow |
| Speaker similarity | resemblyzer cosine score | Lightweight, no GPU needed for eval |
| Dataset format | LJSpeech-compatible CSV | Direct input to CosyVoice 2 fine-tuning scripts |
| Entry point | `python -m speakerforge <stage>` CLI (Click) | Independently runnable per stage; chainable via `run` |

---

## Project Structure

```
SpeakerForge/
├── speakerforge/
│   ├── __init__.py
│   ├── __main__.py              # python -m speakerforge entry
│   ├── cli.py                   # Click group + all subcommands
│   ├── bili_auth.py             # Credential load/save (from Subtitle)
│   ├── config.py                # sources.yaml + pipeline.yaml loaders
│   └── stages/
│       ├── __init__.py
│       ├── stage0_acquire.py    # Resolve + download Bilibili sources
│       ├── stage1_extract.py    # ffmpeg audio extraction → WAV
│       ├── stage2_demucs.py     # Vocal isolation (htdemucs_ft)
│       ├── stage3_vad.py        # VAD segmentation (silero-vad)
│       ├── stage4_align.py      # WhisperX forced alignment
│       ├── stage5_filter.py     # Quality filtering
│       ├── stage6_normalize.py  # RMS normalization + silence trim
│       ├── stage7_package.py    # Dataset packaging (audio/ + metadata.csv)
│       ├── stage8_finetune.py   # CosyVoice 2 LoRA fine-tuning wrapper
│       └── stage9_eval.py       # Inference + resemblyzer speaker similarity
├── sources.yaml.example         # Source whitelist template
├── pipeline.yaml                # Per-stage hyperparameters
├── requirements.txt
└── docs/
    └── superpowers/
        └── specs/
            └── speakerforge-design.md  (this file)
```

---

## CLI Interface

```bash
# One-time login (writes bili_cookies.json)
python -m speakerforge login

# Run individual stage
python -m speakerforge stage0 --speaker up_ai_report
python -m speakerforge stage1 --speaker up_ai_report
python -m speakerforge stage2 --speaker up_ai_report
python -m speakerforge stage3 --speaker up_ai_report
python -m speakerforge stage4 --speaker up_ai_report
python -m speakerforge stage5 --speaker up_ai_report
python -m speakerforge stage6 --speaker up_ai_report
python -m speakerforge stage7 --speaker up_ai_report
python -m speakerforge stage8 --speaker up_ai_report
python -m speakerforge stage9 --speaker up_ai_report

# Chain multiple stages
python -m speakerforge run --stages 0-7 --speaker up_ai_report

# Show dataset stats for a speaker
python -m speakerforge stats --speaker up_ai_report
```

All stage commands accept `--config pipeline.yaml` (default) and `--dry-run` to print what would run.

---

## Data Flow

```
sources.yaml
    │
    ▼ Stage 0 (acquire)
raw_sources/bilibili/<speaker>/
    BVxxxx.mp4
    source_index.json
    │
    ▼ Stage 1 (extract)
processed/<speaker>/audio/
    BVxxxx.wav  (mono, 24kHz)
    │
    ▼ Stage 2 (demucs)
processed/<speaker>/vocals/
    BVxxxx.wav  (vocals only)
    │
    ▼ Stage 3 (vad)
processed/<speaker>/segments/
    BVxxxx_0001.wav
    BVxxxx_0002.wav
    ...
    │
    ▼ Stage 4 (align)
processed/<speaker>/aligned/
    BVxxxx_0001.json  {text, start, end, confidence}
    ...
    │
    ▼ Stage 5 (filter)
processed/<speaker>/filtered/
    manifest.jsonl  (passing segment IDs + metadata)
    │
    ▼ Stage 6 (normalize)
processed/<speaker>/normalized/
    BVxxxx_0001.wav  (RMS normalized, silence trimmed)
    │
    ▼ Stage 7 (package)
dataset/<speaker>/
    audio/0001.wav, 0002.wav, ...
    metadata.csv  (file_path, text, duration, confidence)
    │
    ▼ Stage 8 (finetune)
models/<speaker>/
    cosyvoice2_lora/  (adapter weights)
    train_log.txt
    │
    ▼ Stage 9 (eval)
eval/<speaker>/
    audio/  (generated samples)
    scores.json  {speaker_similarity, prompts: [...]}
```

---

## Stage Specifications

### Stage 0 — Source Acquisition

**Input:** `sources.yaml`

**Source types:**
```yaml
sources:
  - speaker: "up_ai_report"
    platform: bilibili
    type: video
    id: "BV1xxxx"

  - speaker: "up_ai_report"
    platform: bilibili
    type: collection
    id: "xxxx"          # series/collection ID

  - speaker: "up_voice_clean"
    platform: bilibili
    type: user_videos
    uid: "123456"
    limit: 10
```

**Implementation:**
- `bilibili_api.video.Video` for single video metadata
- `bilibili_api.channel_series.ChannelSeries` for collection → BV list
- `bilibili_api.user.User.get_videos()` for user_videos with limit
- `yt-dlp` with `cookiefile=bili_cookies.json` for actual download
- Output: `raw_sources/bilibili/<speaker>/BVxxxx.mp4`
- Index: `raw_sources/bilibili/<speaker>/source_index.json`
- Skip already-downloaded videos (idempotent)

**Constraints:** No auto-search. Only sources explicitly in `sources.yaml`.

---

### Stage 1 — Audio Extraction

**Input:** `raw_sources/bilibili/<speaker>/BVxxxx.mp4`

**Process:**
- `ffmpeg -i input.mp4 -ac 1 -ar 24000 -vn output.wav`
- mono, 24kHz (CosyVoice 2 native sample rate)

**Output:** `processed/<speaker>/audio/BVxxxx.wav`

---

### Stage 2 — Vocal Isolation

**Input:** `processed/<speaker>/audio/BVxxxx.wav`

**Process:**
- `demucs.separate` with model `htdemucs_ft`
- Keep `vocals` stem only
- Re-encode to mono 24kHz WAV after separation

**Output:** `processed/<speaker>/vocals/BVxxxx.wav`

**Note:** Aggressive isolation preferred — slight artifacts < BGM leakage.

---

### Stage 3 — VAD Segmentation

**Input:** `processed/<speaker>/vocals/BVxxxx.wav`

**Process:**
- silero-vad with default thresholds
- Merge segments closer than 0.3s
- Reject segments: duration < 0.8s or > 8s

**Output:** `processed/<speaker>/segments/BVxxxx_<NNN>.wav`

**Target:** Mean segment ~3s; skip silence.

---

### Stage 4 — Transcription + Forced Alignment

**Input:** `processed/<speaker>/segments/BVxxxx_<NNN>.wav`

**Process:**
- WhisperX `large-v3`, `language=zh`, `batch_size=16`
- Word-level forced alignment via phoneme alignment model
- Save per-segment JSON: `{text, start, end, word_segments, confidence}`

**Output:** `processed/<speaker>/aligned/BVxxxx_<NNN>.json`

**Critical:** Use WhisperX alignment output, not raw Whisper timestamps.

---

### Stage 5 — Filtering

**Input:** `processed/<speaker>/aligned/` + `processed/<speaker>/segments/`

**Reject criteria:**
- Duration < 1s or > 6s (post-alignment boundaries)
- Confidence score < 0.6 (if available)
- Text length < 4 characters (likely noise/filler)
- Contains detected overlapping speech (energy variance heuristic)

**Output:** `processed/<speaker>/filtered/manifest.jsonl`
Each line: `{segment_path, text, duration, confidence}`

---

### Stage 6 — Normalization

**Input:** filtered segments (via manifest.jsonl)

**Process:**
- RMS normalization to -20 dBFS
- Trim leading/trailing silence (threshold: -40 dBFS, min 0.1s)
- Enforce: mono, 24kHz, WAV PCM 16-bit

**Output:** `processed/<speaker>/normalized/<seg_id>.wav`

---

### Stage 7 — Dataset Packaging

**Input:** `processed/<speaker>/normalized/` + filtered manifest

**Output:**
```
dataset/<speaker>/
    audio/0001.wav, 0002.wav, ...
    metadata.csv
```

`metadata.csv` columns: `file_path, text, duration, confidence`

Format is LJSpeech-compatible for direct use with CosyVoice 2 fine-tuning scripts.

**Stats printed:** total segments, total duration, duration histogram.

---

### Stage 8 — CosyVoice 2 LoRA Fine-tuning

**Input:** `dataset/<speaker>/`

**Process:**
- Wrap official CosyVoice 2 fine-tuning script (`finetune.py`)
- LoRA only — base model weights frozen
- Default config: batch_size=4, epochs=10, early stopping on val loss
- Checkpoint saved per epoch to `models/<speaker>/cosyvoice2_lora/`

**Constraints:**
- No full fine-tuning
- No aggressive hyperparameter search
- Goal: fast iteration to validate dataset quality

---

### Stage 9 — Inference + Evaluation

**Input:** `models/<speaker>/cosyvoice2_lora/` + test prompts

**Process:**
- Generate speech for 5 fixed test prompts (short + long + unseen)
- Compute resemblyzer cosine similarity vs reference audio (mean of 3 reference clips)
- Output: generated WAVs + `scores.json`

**Evaluation criteria:**
1. Speaker similarity score (resemblyzer cosine, target > 0.75)
2. Subjective audio quality (manual listening)
3. Prosody naturalness (manual listening)

**Experiment comparison:** Run Stage 8–9 for both:
- Version A dataset (Stage 1 → Stage 7, no Demucs/VAD/WhisperX)
- Version B dataset (full pipeline)

---

## Configuration Files

### `sources.yaml`
```yaml
sources:
  - speaker: "speaker_name"
    platform: bilibili
    type: video          # video | collection | user_videos
    id: "BV1xxxx"
```

### `pipeline.yaml`
```yaml
auth:
  cookie_file: "~/.speakerforge/bili_cookies.json"  # override to ../Subtitle/cache/bili_cookies.json to reuse existing session

stage1:
  sample_rate: 24000

stage2:
  model: htdemucs_ft

stage3:
  min_duration: 0.8
  max_duration: 8.0
  min_silence_duration: 0.3

stage4:
  model: large-v3        # requires ≥8GB VRAM; use medium for CPU-only
  language: zh
  batch_size: 16

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
  cosyvoice2_repo: "/path/to/CosyVoice"  # clone https://github.com/FunAudioLLM/CosyVoice
  lora_rank: 16
  epochs: 10
  batch_size: 4
```

---

## Dependencies

```
# Existing in MyTest venv (no install needed)
bilibili-api-python
yt-dlp
pyyaml
click
ffmpeg-python
tqdm

# Need to install
demucs
whisperx
silero-vad
resemblyzer
soundfile
librosa
torch  # if not present
```

CosyVoice 2 must be cloned separately (official repo) and its path configured in `pipeline.yaml`.

---

## Experiment Plan

| | Version A (baseline) | Version B (full pipeline) |
|---|---|---|
| Stages | 0 → 1 → whisper-only segmentation → 7 | 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 |
| Expected quality | Noisy audio, weak alignment | Clean audio, tight alignment |
| LoRA training | Stage 8 on dataset A → `models/<speaker>_versionA/` | Stage 8 on dataset B → `models/<speaker>_versionB/` |
| Evaluation | Stage 9 scores + listening | Stage 9 scores + listening |
