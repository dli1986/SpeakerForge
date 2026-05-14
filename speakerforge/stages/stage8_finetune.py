from __future__ import annotations

import csv
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path


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
    """Run a shell command inside WSL with venv activated, cwd = CosyVoice repo root."""
    inner = f"source '{env_wsl}/bin/activate' && cd '{repo_wsl}' && {cmd}"
    subprocess.run(_wsl(distro, user, "bash", "-c", inner), check=True)


def _build_finetune_yaml(
    distro: str,
    user: str,
    repo_wsl: str,
    out_path: Path,
    epochs: int,
) -> None:
    """Read cosyvoice2.yaml from WSL, patch for fine-tuning, write to out_path."""
    src = f"{repo_wsl}/examples/libritts/cosyvoice2/conf/cosyvoice2.yaml"
    result = subprocess.run(
        _wsl(distro, user, "cat", src),
        capture_output=True, text=True, check=True,
    )
    yaml_text = result.stdout
    yaml_text = re.sub(r"(max_epoch:\s*)\d+", rf"\g<1>{epochs}", yaml_text)
    yaml_text = re.sub(r"(use_spk_embedding:\s*)False", r"\g<1>True", yaml_text)
    out_path.write_text(yaml_text, encoding="utf-8")


def _prepare_kaldi_data(metadata_csv: Path, data_out: Path, speaker: str) -> None:
    """Create wav.scp / text / utt2spk / spk2utt from stage7 metadata.csv."""
    data_out.mkdir(parents=True, exist_ok=True)
    src_wavs = metadata_csv.parent / "wavs"

    wav_scp, text_lines, utt2spk = [], [], []
    spk2utt: dict[str, list[str]] = defaultdict(list)

    with open(metadata_csv, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            utt_id = row["filename"]
            wav_wsl = _win_to_wsl(src_wavs / (utt_id + ".wav"))
            wav_scp.append(f"{utt_id} {wav_wsl}")
            text_lines.append(f"{utt_id} {row['text']}")
            utt2spk.append(f"{utt_id} {speaker}")
            spk2utt[speaker].append(utt_id)

    (data_out / "wav.scp").write_text("\n".join(sorted(wav_scp)) + "\n", encoding="utf-8")
    (data_out / "text").write_text("\n".join(sorted(text_lines)) + "\n", encoding="utf-8")
    (data_out / "utt2spk").write_text("\n".join(sorted(utt2spk)) + "\n", encoding="utf-8")
    spk2utt_lines = [
        f"{spk} {' '.join(sorted(utts))}" for spk, utts in spk2utt.items()
    ]
    (data_out / "spk2utt").write_text("\n".join(spk2utt_lines) + "\n", encoding="utf-8")


def run(pipeline_cfg, speaker: str, version: str = "B") -> None:
    cfg = pipeline_cfg.stage8
    repo_str = cfg.get("cosyvoice2_repo", "")
    if not repo_str or repo_str == "/path/to/CosyVoice":
        raise ValueError(
            "stage8.cosyvoice2_repo is not configured in pipeline.yaml. "
            "Clone CosyVoice2 and set the path there."
        )

    repo_wsl = _to_wsl_str(repo_str)
    env_wsl = _to_wsl_str(cfg.get("cosyvoice_env", "/home/dli/cosyvoice_env"))
    distro = cfg.get("wsl_distro", "Ubuntu")
    user = cfg.get("wsl_user", "dli")
    pretrained = cfg.get("pretrained_model_dir", "pretrained_models/CosyVoice2-0.5B")
    epochs = cfg.get("epochs", 10)

    tag = f"{speaker}_version{version}"
    metadata_csv = Path("dataset") / tag / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(
            f"Stage7 output not found: {metadata_csv}. Run stage7 first."
        )

    data_dir = Path("cosyvoice_data") / tag
    output_dir = Path("models") / tag

    # Clean both dirs before every run to prevent stale checkpoints/embeddings
    if data_dir.exists():
        shutil.rmtree(data_dir)
        print(f"[stage8] Cleaned: {data_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
        print(f"[stage8] Cleaned: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    data_wsl = _win_to_wsl(data_dir)
    out_wsl = _win_to_wsl(output_dir)

    # 1. Kaldi data files
    print("[stage8] Preparing Kaldi data files...")
    _prepare_kaldi_data(metadata_csv, data_dir, speaker)

    # 2. Fine-tune yaml (max_epoch + use_spk_embedding patched)
    print("[stage8] Generating fine-tune config...")
    finetune_yaml = data_dir / "cosyvoice2_finetune.yaml"
    _build_finetune_yaml(distro, user, repo_wsl, finetune_yaml, epochs)
    finetune_yaml_wsl = _win_to_wsl(finetune_yaml)

    # 3. Speaker embeddings
    print("[stage8] Extracting speaker embeddings...")
    _wsl_run(distro, user, repo_wsl, env_wsl,
        f"python tools/extract_embedding.py"
        f" --dir '{data_wsl}'"
        f" --onnx_path '{pretrained}/campplus.onnx'"
    )

    # 4. Speech tokens
    print("[stage8] Extracting speech tokens...")
    _wsl_run(distro, user, repo_wsl, env_wsl,
        f"python tools/extract_speech_token.py"
        f" --dir '{data_wsl}'"
        f" --onnx_path '{pretrained}/speech_tokenizer_v2.onnx'"
    )

    # 5. Parquet dataset
    print("[stage8] Building parquet dataset...")
    (data_dir / "parquet").mkdir(parents=True, exist_ok=True)
    _wsl_run(distro, user, repo_wsl, env_wsl,
        f"python tools/make_parquet_list.py"
        f" --num_utts_per_parquet 50"
        f" --num_processes 1"
        f" --src_dir '{data_wsl}'"
        f" --des_dir '{data_wsl}/parquet'"
    )

    # 6. Fine-tune LLM
    print("[stage8] Fine-tuning LLM...")
    _wsl_run(distro, user, repo_wsl, env_wsl,
        f"PYTHONPATH=. torchrun --nnodes=1 --nproc_per_node=1"
        f" --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:1234"
        f" cosyvoice/bin/train.py"
        f" --config '{finetune_yaml_wsl}'"
        f" --train_data '{data_wsl}/parquet/data.list'"
        f" --cv_data '{data_wsl}/parquet/data.list'"
        f" --qwen_pretrain_path '{pretrained}/CosyVoice-BlankEN'"
        f" --onnx_path '{pretrained}'"
        f" --model llm"
        f" --checkpoint '{pretrained}/llm.pt'"
        f" --model_dir '{out_wsl}'"
        f" --num_workers 1"
        f" --prefetch 10"
        f" --pin_memory"
    )
