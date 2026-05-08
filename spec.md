🧠 Project: Lightweight Voice Cloning Dataset Pipeline (LoRA-ready)
🎯 Goal
Build a reproducible, modular pipeline that converts clean or semi-clean online video/audio sources (e.g., Bilibili videos) into a high-quality speech dataset suitable for LoRA-based speaker adaptation on small conversational TTS models (e.g., CSM-1B).
The goal is not full-scale production, but an experimental, personal-use pipeline that emphasizes data quality over quantity.

🧱 Core Output Format
The final dataset must consist of aligned pairs:
(audio_segment.wav, transcript.txt)

With constraints:

Audio:

mono
16kHz or 24kHz
duration: 1–6 seconds (target ~3 seconds)


Text:

accurately aligned to audio
minimal transcription errors


Speaker:

single speaker per dataset


No background music or overlapping speakers


⚙️ Pipeline Overview
The system should implement the following stages:

🧱 Stage 0 — Bilibili Source Acquisition (Config + Existing Auth Integration)
🎯 Goal
Leverage existing Bilibili authentication and downloader capabilities to programmatically acquire high-quality video/audio sources, while maintaining strict human control over source selection.
This stage is tightly integrated with existing internal tools (e.g., Subtitle project, QR-code auth).

✅ Core Design Principles

Human defines which sources (UP, series, video)
Program automates resolution, download, and organization
Reuse existing tooling whenever possible


🔐 Authentication (Reuse Existing Implementation)

Use existing Bilibili QR-code authentication
Reuse session/cookies from your other project (e.g., Subtitle)
Agent should NOT reimplement auth unless necessary


📥 Input — Config-driven (Required)
All sources must be defined explicitly via config (e.g., sources.yaml):

Example
sources:
  - speaker: "up_ai_report"
    platform: bilibili
    type: collection
    id: "xxxx"

  - speaker: "up_ai_report"
    platform: bilibili
    type: video
    id: "BV1xxxx"

  - speaker: "up_voice_clean"
    platform: bilibili
    type: user_videos
    uid: "123456"
    limit: 10

✅ Supported Source Types


TypeDescriptionvideo单个视频collection / playlist系列user_videos指定UP主视频（需limit）

⚙️ Processing Flow
sources.yaml
↓
Resolver:
  - expand collection → video list
  - expand user_videos (with limit)
↓
Downloader:
  - call bilibili API / existing library
↓
Store raw video
↓
Pass to Stage 1


📦 Output Structure
raw_sources/
  bilibili/
    <speaker>/
      BVxxxx.mp4
      BVyyyy.mp4


🔁 Integration Requirement (IMPORTANT)
Agent must:

reuse existing Subtitle project utilities if possible
reuse bili auth module (qrcode login result)
stay within MyTest virtual environment


🔧 Download Behavior
Audio Extraction Strategy
Either:

download video → extract audio later (recommended)

OR

directly extract audio stream if supported


⚠️ Key Constraints
1️⃣ Human-defined White List (MANDATORY)

No auto-search / no crawling
Only process sources explicitly listed in config


2️⃣ Speaker Consistency

Each speaker group must represent one voice identity
Do NOT mix speakers in one dataset


3️⃣ Limit Scale (Important for Experiment)

Prefer small, high-quality datasets
Target: 1–2 hours total audio per speaker


4️⃣ Avoid Over-Automation
Even though API is available:

Do NOT auto-expand entire UP video history
Must support limit or manual filtering


⚠️ Key Engineering Concerns
API / Auth Stability

Bilibili APIs may change
Must support fallback via existing working library


Format Inconsistency

video codecs / audio streams vary
normalize downstream (Stage 1)


Subtitle Reliability (Important Note)

Bilibili subtitles/danmaku are NOT reliable for TTS alignment
Should NOT be used directly for dataset creation


📊 Optional Metadata Index (Recommended)
Create:
source_index.json

Example:
{
  "BV1xxxx": {
    "speaker": "up_ai_report",
    "duration": 320,
    "status": "downloaded"
  }
}

🚧 Non-goals

No YouTube support (handled in other project)
No general crawler
No recommendation-based input


✅ Expected Outcome

reproducible, config-driven dataset sourcing
seamless reuse of existing Bilibili tooling
aligned with downstream speech dataset pipeline

Stage 1 — Source Acquisition
Input:

local video/audio files (manually downloaded or provided)

Process:

extract audio via ffmpeg
convert to .wav

Constraints:

normalize format early (sample rate, mono channel)


Stage 2 — Vocal Isolation (Critical)
Goal:

remove background music and noise

Suggested:

Demucs or equivalent source separation

Key Concern:

residual BGM will degrade model quality
acceptable tradeoff: slight artifacts > background music leakage


Stage 3 — Pre-segmentation (VAD)
Goal:

split long audio into speech segments

Approach:

use Voice Activity Detection (VAD)

Constraints:

remove silence
avoid overly long segments (>8s)
avoid overly short segments (<0.8s)


Stage 4 — Transcription + Alignment (Critical)
Goal:

produce accurately aligned (audio, text) pairs

Approach:

Use WhisperX (preferred) or Whisper + forced alignment

Important:

Do NOT rely on Whisper raw segmentation alone
Must include time-aligned word/segment boundaries

Key Concerns:

punctuation errors acceptable
timing misalignment NOT acceptable


Stage 5 — Post-processing / Filtering
The system must filter out low-quality samples:
Remove segments that:

are too short (<1s)
are too long (>6–8s)
contain long silence
have low ASR confidence (if available)
contain overlapping speech
include music/noise artifacts

Optional:

simple text normalization (lowercase, cleanup)
remove filler-heavy sentences


Stage 6 — Normalization
Normalize all audio:

consistent volume (RMS normalization)
consistent format (wav, mono, 16kHz/24kHz)
trim leading/trailing silence


Stage 7 — Dataset Packaging
Output structure:
dataset/
  ├── audio/
  │     ├── 0001.wav
  │     ├── 0002.wav
  ├── metadata.csv / jsonl

Metadata fields:
file_path, text

Optional:
duration, confidence_score


🔥 Key Engineering Concerns (must prioritize)
1. Alignment Quality > Transcription Accuracy

Slight text errors are acceptable
Timing mismatch is NOT acceptable
Model learns rhythm from alignment


2. Audio Purity is Critical

Background music strongly degrades results
Always prefer aggressive vocal isolation


3. Segment Length Distribution

Avoid very long segments → harms learning
Avoid very short segments → loses prosody

Target:

mean ~3 seconds


4. Dataset Size vs Quality

30–60 minutes of clean data is sufficient for initial LoRA experiments
Do NOT optimize for scale yet


🧪 Experiment Plan (important)
The agent should support running two dataset versions:

Version A (baseline)
ffmpeg → Whisper segmentation → dataset


Version B (enhanced)
ffmpeg → Demucs → VAD → WhisperX → filtering → dataset


Evaluation
Compare:

audio clarity
speaker similarity
prosody stability


🧩 Modularity Requirements
The pipeline should be modular:

each stage independently runnable
intermediate outputs cached
easy to swap components (e.g., Demucs ↔ other models)


🚧 Non-goals

No need for UI
No need for real-time processing
No multi-speaker support
No training implementation (dataset-only project)


✅ Expected Outcome
A reproducible dataset-building pipeline that produces:

LoRA-ready speech datasets
measurable improvement between naive vs cleaned pipeline
usable for small-scale voice cloning experiments

🧱 Stage 8 — LoRA Fine-tuning (Validation Stage)
🎯 Goal
Validate dataset quality by fine-tuning a lightweight LoRA adapter on a base conversational speech model (e.g., CSM-1B).
This stage is intended for evaluation, not optimization.

✅ Model Choice

Base model: CSM-1B (or equivalent conversational TTS model)
Fine-tuning method: LoRA (parameter-efficient)


📥 Input

dataset generated from Stage 7
single-speaker dataset


⚙️ Training Strategy

Freeze base model weights
Train only LoRA layers


✅ Dataset Size Expectation

30–60 minutes minimum
ideal: 1–2 hours clean data


🔧 Training Configuration (Guideline)

batch size: small (based on GPU)
epochs: limited (e.g., 3–10)
early stopping allowed


⚠️ Key Constraints
1️⃣ No full fine-tuning

Only LoRA allowed
Avoid high compute cost


2️⃣ No aggressive hyperparameter tuning

keep default / simple config


3️⃣ Focus on fast iteration
Goal:
dataset version → train → evaluate → iterate



🧱 Stage 9 — Inference & Evaluation
🎯 Goal
Evaluate whether the dataset produces a usable and coherent cloned voice.

✅ Inference Inputs

trained LoRA adapter
test text prompts


Suggested prompts:

short sentences
long paragraphs
unseen content


✅ Evaluation Criteria
1️⃣ Speaker Similarity

does it sound like target speaker?


2️⃣ Audio Quality

clean vs noisy
artifacts present?


3️⃣ Prosody / Naturalness

correct pauses?
natural rhythm?


4️⃣ Stability

consistent across sentences?
degradation in long text?



🧪 Experiment Design (Critical)
Compare:

Dataset A (baseline)
no demucs / no alignment


Dataset B (enhanced)
full pipeline


🔍 Outcome

qualitative comparison (listening)
note failure modes

          你的核心资产
     ✅ 高质量语音数据集
               ↓
      ┌────────────┐
      ↓            ↓
CosyVoice       CSM
(验证)          (目标模型)
↓               ↓
是否像 ✅     对话语音 ✅