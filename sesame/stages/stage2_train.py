from __future__ import annotations

import torch
from pathlib import Path

from sesame.config import SesameConfig


def _preprocess_example(example: dict, processor, speaker_id: str = "0") -> dict | None:
    conversation = [
        {
            "role": speaker_id,
            "content": [
                {"type": "text", "text": example["text"]},
                {"type": "audio", "path": example["audio"]["array"]},
            ],
        }
    ]
    try:
        model_inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            output_labels=True,
            text_kwargs={
                "padding": "max_length",
                "max_length": 256,
                "pad_to_multiple_of": 8,
                "padding_side": "right",
            },
            audio_kwargs={
                "sampling_rate": 24_000,
                "max_length": 240001,
                "padding": "max_length",
            },
            common_kwargs={"return_tensors": "pt"},
        )
    except Exception as e:
        print(f"  [WARN] Skipping example '{example['text'][:40]}': {e}")
        return None

    required = ["input_ids", "attention_mask", "labels", "input_values", "input_values_cutoffs"]
    result = {}
    for key in required:
        if key not in model_inputs:
            print(f"  [WARN] Missing key '{key}', skipping example.")
            return None
        result[key] = model_inputs[key][0]

    if not all(isinstance(result[k], torch.Tensor) for k in result):
        return None
    return result


def run(cfg: SesameConfig) -> None:
    from unsloth import FastModel, is_bfloat16_supported
    from transformers import CsmForConditionalGeneration, AutoProcessor, Trainer, TrainingArguments
    from datasets import load_from_disk, Audio

    speaker = cfg.dataset.get("speaker", "Akinokoe")
    version = cfg.dataset.get("version", "B")

    data_dir = Path("data") / f"{speaker}_v{version}"
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset not found: {data_dir}. Run stage1 first.")

    model_name = cfg.model.get("name", "unsloth/csm-1b")
    max_seq_length = cfg.model.get("max_seq_length", 2048)
    load_in_4bit = cfg.model.get("load_in_4bit", False)

    print(f"[stage2] Loading model: {model_name}  load_in_4bit={load_in_4bit}")
    model, _ = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        auto_model=CsmForConditionalGeneration,
        load_in_4bit=load_in_4bit,
    )

    lora_r = cfg.lora.get("r", 32)
    lora_alpha = cfg.lora.get("alpha", 32)
    lora_dropout = cfg.lora.get("dropout", 0.0)
    target_modules = cfg.lora.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    print(f"[stage2] Applying LoRA: r={lora_r}, alpha={lora_alpha}, targets={target_modules}")
    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    processor = AutoProcessor.from_pretrained(model_name)

    print(f"[stage2] Loading + preprocessing dataset from {data_dir}...")
    raw_ds = load_from_disk(str(data_dir))
    raw_ds = raw_ds.cast_column("audio", Audio(sampling_rate=24000))

    # Add speaker_id column (single speaker → always 0)
    if "source" not in raw_ds.column_names:
        raw_ds = raw_ds.add_column("source", ["0"] * len(raw_ds))

    processed_ds = raw_ds.map(
        lambda ex: _preprocess_example(ex, processor, speaker_id="0"),
        remove_columns=raw_ds.column_names,
        desc="Preprocessing",
    )
    processed_ds = processed_ds.filter(lambda ex: ex is not None)
    print(f"[stage2] Processed {len(processed_ds)} samples (dropped malformed)")

    output_dir = cfg.training.get("output_dir", f"models/{speaker}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        model=model,
        train_dataset=processed_ds,
        args=TrainingArguments(
            per_device_train_batch_size=cfg.training.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=cfg.training.get("gradient_accumulation_steps", 4),
            warmup_steps=cfg.training.get("warmup_steps", 5),
            max_steps=cfg.training.get("max_steps", 60),
            learning_rate=cfg.training.get("learning_rate", 2e-4),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=cfg.training.get("max_steps", 60),  # save once at end
            report_to="none",
        ),
    )

    print("[stage2] Starting training...")
    stats = trainer.train()
    runtime = stats.metrics.get("train_runtime", 0)
    print(f"[stage2] Done. Runtime: {runtime:.0f}s ({runtime/60:.1f}min)")

    adapter_path = Path(output_dir) / "lora_adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"[stage2] Adapter saved → {adapter_path}")
