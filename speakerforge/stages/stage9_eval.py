from __future__ import annotations

import csv
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np

from speakerforge.config import PipelineConfig


def _win_to_wsl(path: Path) -> str:
    p = path.resolve()
    drive = p.drive.rstrip(":").lower()
    rest = str(p)[len(p.drive):].replace("\\", "/")
    return f"/mnt/{drive}{rest}"


def _to_wsl_str(path_or_str: str | Path) -> str:
    s = str(path_or_str)
    if s.startswith("/"):
        return s
    return _win_to_wsl(Path(s))


def _wsl(distro: str, user: str, *args: str) -> list[str]:
    return ["wsl", "-d", distro, "-u", user] + list(args)


def _wsl_run(distro: str, user: str, repo_wsl: str, env_wsl: str, cmd: str) -> None:
    inner = f"source '{env_wsl}/bin/activate' && cd '{repo_wsl}' && {cmd}"
    subprocess.run(_wsl(distro, user, "bash", "-c", inner), check=True)


def _prepare_merged_model(
    distro: str, user: str, repo_wsl: str, env_wsl: str,
    pretrained_wsl: str, best_ckpt_wsl: str, merged_dir_wsl: str,
) -> None:
    """Copy pretrained model files into merged_dir, replace llm.pt with fine-tuned ckpt."""
    cmd = (
        f"mkdir -p '{merged_dir_wsl}' && "
        f"cp '{pretrained_wsl}'/*.onnx '{merged_dir_wsl}/' && "
        f"cp '{pretrained_wsl}'/*.yaml '{merged_dir_wsl}/' && "
        f"cp '{pretrained_wsl}'/*.json '{merged_dir_wsl}/' && "
        f"cp '{pretrained_wsl}'/flow.pt '{merged_dir_wsl}/' && "
        f"cp '{pretrained_wsl}'/hift.pt '{merged_dir_wsl}/' && "
        f"cp -r '{pretrained_wsl}'/CosyVoice-BlankEN '{merged_dir_wsl}/' && "
        f"cp '{best_ckpt_wsl}' '{merged_dir_wsl}/llm.pt'"
    )
    _wsl_run(distro, user, repo_wsl, env_wsl, cmd)

    # Strip training-only keys ("epoch", "step") that load_state_dict(strict=True) rejects
    strip_cmd = (
        f"python -c \""
        f"import torch; "
        f"d = torch.load('{merged_dir_wsl}/llm.pt', map_location='cpu', weights_only=False); "
        f"[d.pop(k, None) for k in ['epoch', 'step']]; "
        f"torch.save(d, '{merged_dir_wsl}/llm.pt')"
        f"\""
    )
    _wsl_run(distro, user, repo_wsl, env_wsl, strip_cmd)


def _run_inference_wsl(
    distro: str, user: str, repo_wsl: str, env_wsl: str,
    merged_model_wsl: str, pretrained_model_wsl: str, ref_wav_wsl: str, ref_text: str,
    texts: list[str], out_dir_wsl: str, script_path_wsl: str,
    speaker: str = "Akinokoe",
) -> None:
    sft_dir_wsl = out_dir_wsl.rstrip("/") + "_sft"
    baseline_dir_wsl = out_dir_wsl.rstrip("/") + "_baseline"
    script = textwrap.dedent(f"""\
        import os, sys
        sys.path.insert(0, '.')
        sys.path.insert(0, 'third_party/Matcha-TTS')
        from cosyvoice.cli.cosyvoice import CosyVoice2
        import torchaudio

        finetuned = CosyVoice2('{merged_model_wsl}')
        pretrained = CosyVoice2('{pretrained_model_wsl}')
        ref_text = {repr(ref_text)}
        ref_wav = '{ref_wav_wsl}'
        texts = {repr(texts)}

        print('=== zero_shot (finetuned) ===')
        os.makedirs('{out_dir_wsl}', exist_ok=True)
        for i, text in enumerate(texts):
            out_wav = '{out_dir_wsl}/gen_{{:04d}}.wav'.format(i)
            for result in finetuned.inference_zero_shot(text, ref_text, ref_wav, stream=False):
                torchaudio.save(out_wav, result['tts_speech'], finetuned.sample_rate)
                break
            print(f'Generated {{i+1}}/{{len(texts)}}: {{out_wav}}')

        print('=== inference_sft (finetuned) ===')
        os.makedirs('{sft_dir_wsl}', exist_ok=True)
        for i, text in enumerate(texts):
            out_wav = '{sft_dir_wsl}/gen_{{:04d}}.wav'.format(i)
            for result in finetuned.inference_sft(text, '{speaker}', stream=False):
                torchaudio.save(out_wav, result['tts_speech'], finetuned.sample_rate)
                break
            print(f'Generated {{i+1}}/{{len(texts)}}: {{out_wav}}')

        print('=== zero_shot (pretrained baseline) ===')
        os.makedirs('{baseline_dir_wsl}', exist_ok=True)
        for i, text in enumerate(texts):
            out_wav = '{baseline_dir_wsl}/gen_{{:04d}}.wav'.format(i)
            for result in pretrained.inference_zero_shot(text, ref_text, ref_wav, stream=False):
                torchaudio.save(out_wav, result['tts_speech'], pretrained.sample_rate)
                break
            print(f'Generated {{i+1}}/{{len(texts)}}: {{out_wav}}')
    """)

    # Write script to a Windows path accessible from WSL
    script_local = Path(script_path_wsl.replace("/mnt/c/", "C:/").replace("/", "\\")) \
        if script_path_wsl.startswith("/mnt/c/") else None
    if script_local:
        script_local.write_text(script, encoding="utf-8")
    else:
        _wsl_run(distro, user, repo_wsl, env_wsl,
                 f"printf '%s' {repr(script)} > '{script_path_wsl}'")

    _wsl_run(distro, user, repo_wsl, env_wsl,
             f"PYTHONPATH=. python '{script_path_wsl}'")


def _pick_best_checkpoint(model_dir: Path) -> Path:
    """Return the last epoch_*_whole.pt checkpoint."""
    ckpts = sorted(model_dir.glob("epoch_*_whole.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"No epoch_*_whole.pt found in {model_dir}")
    return ckpts[-1]


def _inject_speaker_embedding(data_dir: Path, merged_dir: Path, speaker: str) -> None:
    """Create spk2info.pt in merged_model from the training-time speaker embedding."""
    import torch
    spk2emb_path = data_dir / "spk2embedding.pt"
    if not spk2emb_path.exists():
        raise FileNotFoundError(f"spk2embedding.pt not found at {spk2emb_path}. Run stage8 first.")
    spk2emb = torch.load(str(spk2emb_path), map_location="cpu", weights_only=False)
    raw = spk2emb.get(speaker)
    if raw is None:
        raise KeyError(f"Speaker '{speaker}' not found in spk2embedding.pt (keys: {list(spk2emb.keys())})")
    embedding = torch.tensor(raw) if isinstance(raw, list) else raw
    if embedding.dim() == 1:
        embedding = embedding.unsqueeze(0)  # [192] → [1, 192] as flow expects
    spk2info = {speaker: {"embedding": embedding}}
    out_path = merged_dir / "spk2info.pt"
    torch.save(spk2info, str(out_path))
    print(f"[stage9] spk2info.pt written: {speaker} embedding shape={embedding.shape}")


def _collect_reference_wavs(dataset_dir: Path, n: int = 20) -> list[Path]:
    return sorted((dataset_dir / "wavs").glob("*.wav"))[:n]


def _load_test_texts(dataset_dir: Path, n: int = 5) -> tuple[list[str], list[str]]:
    """Return (filenames, texts) for first n rows of metadata.csv."""
    csv_path = dataset_dir / "metadata.csv"
    filenames, texts = [], []
    with open(csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            filenames.append(row["filename"])
            texts.append(row["text"])
            if len(texts) >= n:
                break
    return filenames, texts


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


def run(pipeline_cfg: PipelineConfig, speaker: str, version: str = "B") -> None:
    cfg = pipeline_cfg.stage8
    repo_str = cfg.get("cosyvoice2_repo", "")
    if not repo_str or repo_str == "/path/to/CosyVoice":
        raise ValueError("stage8.cosyvoice2_repo is not set in pipeline.yaml.")

    repo_wsl = _to_wsl_str(repo_str)
    env_wsl = _to_wsl_str(cfg.get("cosyvoice_env", "/home/dli/cosyvoice_env"))
    distro = cfg.get("wsl_distro", "Ubuntu")
    user = cfg.get("wsl_user", "dli")
    pretrained = cfg.get("pretrained_model_dir", "pretrained_models/CosyVoice2-0.5B")
    pretrained_wsl = f"{repo_wsl}/{pretrained}"

    tag = f"{speaker}_version{version}"
    dataset_dir = Path("dataset") / tag
    model_dir = Path("models") / tag
    data_dir = Path("cosyvoice_data") / tag

    if not model_dir.exists():
        raise FileNotFoundError(f"No fine-tuned model at {model_dir}. Run stage8 first.")

    # Locate best checkpoint
    # Locate checkpoint
    ckpt_name = pipeline_cfg.stage9.get("checkpoint", "")
    if ckpt_name:
        best_ckpt = model_dir / ckpt_name
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {best_ckpt}")
    else:
        best_ckpt = _pick_best_checkpoint(model_dir)
    print(f"[stage9] Using checkpoint: {best_ckpt.name}")

    # Prepare merged model dir (pretrained structure + fine-tuned llm.pt)
    merged_dir = model_dir / "merged_model"
    merged_dir_wsl = _win_to_wsl(merged_dir)
    best_ckpt_wsl = _win_to_wsl(best_ckpt)
    print("[stage9] Building merged model directory...")
    _prepare_merged_model(distro, user, repo_wsl, env_wsl,
                          pretrained_wsl, best_ckpt_wsl, merged_dir_wsl)
    _inject_speaker_embedding(data_dir, merged_dir, speaker)

    # Reference wav and text (first sample from dataset)
    ref_wavs = _collect_reference_wavs(dataset_dir)
    if not ref_wavs:
        raise FileNotFoundError(f"No reference WAVs in {dataset_dir / 'wavs'}")

    # Build filename→text lookup from metadata.csv
    import csv as _csv
    text_by_stem: dict[str, str] = {}
    with open(dataset_dir / "metadata.csv", encoding="utf-8-sig") as _f:
        for _row in _csv.DictReader(_f):
            text_by_stem[_row["filename"]] = _row["text"]

    # Use configured ref_wav if specified, otherwise use first dataset wav
    cfg9_ref = pipeline_cfg.stage9.get("ref_wav", "")
    if cfg9_ref:
        ref_wav_path = Path(cfg9_ref)
        if not ref_wav_path.exists():
            raise FileNotFoundError(f"stage9.ref_wav not found: {ref_wav_path}")
        ref_wav_wsl = _win_to_wsl(ref_wav_path)
        ref_text = text_by_stem.get(ref_wav_path.stem, "")
        print(f"[stage9] Using configured ref_wav: {ref_wav_path.name}  ref_text: {ref_text[:30]}...")
    else:
        ref_wav_wsl = _win_to_wsl(ref_wavs[0])
        ref_text = text_by_stem.get(ref_wavs[0].stem, "")
        print(f"[stage9] Using auto ref_wav: {ref_wavs[0].name}  ref_text: {ref_text[:30]}...")

    # Test texts for generation (next 5 after the reference)
    # Test texts: prefer pipeline.yaml stage9.test_texts, fall back to metadata.csv
    cfg9 = pipeline_cfg.stage9
    if cfg9.get("test_texts"):
        test_texts = cfg9["test_texts"]
    else:
        _, test_texts = _load_test_texts(dataset_dir, n=6)
        test_texts = test_texts[1:]  # skip the one used as reference prompt

    # Output directory
    gen_dir = Path("eval") / tag / "generated"
    for d in [gen_dir,
              gen_dir.parent / "generated_sft",
              gen_dir.parent / "generated_baseline"]:
        if d.exists():
            shutil.rmtree(d)
    gen_dir.mkdir(parents=True, exist_ok=True)
    gen_dir_wsl = _win_to_wsl(gen_dir)

    # Inference script path
    script_path = gen_dir / "infer.py"
    script_path_wsl = _win_to_wsl(script_path)

    print("[stage9] Running inference...")
    _run_inference_wsl(distro, user, repo_wsl, env_wsl,
                       merged_model_wsl=merged_dir_wsl,
                       pretrained_model_wsl=pretrained_wsl,
                       ref_wav_wsl=ref_wav_wsl,
                       ref_text=ref_text,
                       texts=test_texts,
                       out_dir_wsl=gen_dir_wsl,
                       script_path_wsl=script_path_wsl,
                       speaker=speaker)

    gen_wavs = sorted(gen_dir.glob("*.wav"))
    if not gen_wavs:
        print("[WARN] No generated WAVs found — skipping similarity scoring.")
        return

    print("[stage9] Computing speaker similarity...")
    try:
        scores = _compute_similarity(ref_wavs, gen_wavs)
        mean_score = float(np.mean(scores))
        print(f"\nSpeaker similarity (resemblyzer cosine):")
        print(f"  mean={mean_score:.4f}  min={min(scores):.4f}  max={max(scores):.4f}")

        report = {
            "speaker": speaker, "version": version,
            "checkpoint": best_ckpt.name,
            "n_generated": len(gen_wavs), "n_reference": len(ref_wavs),
            "similarity_mean": mean_score,
            "similarity_min": float(min(scores)),
            "similarity_max": float(max(scores)),
            "scores": scores,
        }
        report_path = Path("eval") / tag / "similarity_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report saved to {report_path}")
    except ImportError:
        print("[WARN] resemblyzer not installed — skipping similarity scoring.")
        print("       Run: pip install resemblyzer")
