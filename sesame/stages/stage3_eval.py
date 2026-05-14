from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from sesame.config import SesameConfig


def _compute_similarity(ref_wavs: list[Path], gen_wavs: list[Path]) -> list[float]:
    from resemblyzer import VoiceEncoder, preprocess_wav

    encoder = VoiceEncoder()
    ref_embeds = []
    for wav in ref_wavs:
        try:
            ref_embeds.append(encoder.embed_utterance(preprocess_wav(wav)))
        except Exception as exc:
            print(f"  [WARN] Could not embed reference {wav.name}: {exc}")

    if not ref_embeds:
        raise RuntimeError("No reference embeddings could be computed.")

    ref_mean = np.mean(ref_embeds, axis=0)
    ref_mean /= np.linalg.norm(ref_mean)

    scores = []
    for wav in gen_wavs:
        try:
            gen_embed = encoder.embed_utterance(preprocess_wav(wav))
            gen_embed /= np.linalg.norm(gen_embed)
            scores.append(float(np.dot(ref_mean, gen_embed)))
        except Exception as exc:
            print(f"  [WARN] Could not embed generated {wav.name}: {exc}")

    return scores


def run(cfg: SesameConfig) -> None:
    import torch
    import torchaudio
    from unsloth import FastModel
    from transformers import CsmForConditionalGeneration, AutoProcessor

    speaker = cfg.dataset.get("speaker", "Akinokoe")
    version = cfg.dataset.get("version", "B")
    dataset_root = Path(cfg.dataset.get("root", "../speakerforge/dataset"))

    adapter_path = Path(cfg.training.get("output_dir", f"models/{speaker}")) / "lora_adapter"
    if not adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}. Run stage2 first.")

    model_name = str(adapter_path)
    max_seq_length = cfg.model.get("max_seq_length", 2048)

    print(f"[stage3] Loading model + adapter from {adapter_path}...")
    model, _ = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        auto_model=CsmForConditionalGeneration,
        load_in_4bit=False,
    )
    FastModel.for_inference(model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    device = next(model.parameters()).device

    processor = AutoProcessor.from_pretrained(model_name)
    sample_rate = 24000
    speaker_id = 0

    eval_dir = Path("eval") / speaker / "generated"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    test_texts: list[str] = cfg.eval.get("test_texts", [])
    if not test_texts:
        print("[WARN] No test_texts in config — skipping generation.")
        return

    print(f"[stage3] Generating {len(test_texts)} samples...")
    for i, text in enumerate(test_texts):
        inputs = processor(
            f"[{speaker_id}]{text}",
            add_special_tokens=True,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            audio_values = model.generate(
                **inputs,
                max_new_tokens=cfg.eval.get("max_new_tokens", 500),  # ~40s ceiling
            )

        # audio_values shape: (1, T) float waveform
        wav_tensor = audio_values.cpu().float()
        if wav_tensor.dim() == 1:
            wav_tensor = wav_tensor.unsqueeze(0)

        out_wav = eval_dir / f"gen_{i:04d}.wav"
        torchaudio.save(str(out_wav), wav_tensor, sample_rate)
        print(f"  Generated {i+1}/{len(test_texts)}: {out_wav.name}")

    gen_wavs = sorted(eval_dir.glob("*.wav"))
    if not gen_wavs:
        print("[WARN] No generated WAVs found — skipping similarity scoring.")
        return

    n_ref = cfg.eval.get("n_ref_wavs", 20)
    ref_wavs = sorted((dataset_root / f"{speaker}_version{version}" / "wavs").glob("*.wav"))[:n_ref]
    if not ref_wavs:
        print("[WARN] No reference WAVs found — skipping similarity scoring.")
        return

    print("[stage3] Computing speaker similarity...")
    try:
        scores = _compute_similarity(ref_wavs, gen_wavs)
        mean_score = float(np.mean(scores))
        print(f"\nSpeaker similarity (resemblyzer cosine):")
        print(f"  mean={mean_score:.4f}  min={min(scores):.4f}  max={max(scores):.4f}")

        report = {
            "speaker": speaker,
            "adapter": str(adapter_path),
            "n_generated": len(gen_wavs),
            "n_reference": len(ref_wavs),
            "similarity_mean": mean_score,
            "similarity_min": float(min(scores)),
            "similarity_max": float(max(scores)),
            "scores": scores,
        }
        report_path = Path("eval") / speaker / "similarity_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report saved → {report_path}")
    except ImportError:
        print("[WARN] resemblyzer not installed — skipping similarity scoring.")
        print("       Run: pip install resemblyzer")
