Objective:
Train a LoRA adapter on CSM-1B to inject a single speaker's voice and speaking style,
without altering the model's inherent dialogue capabilities.

Data:
- Single-speaker audio-text dataset (600–1000 samples)
- Format: {"audio": path, "text": string}
- No dialogue structure required
- No multi-speaker mixing

Training Strategy:
- LoRA only (freeze base model)
- Do NOT perform full SFT
- Do NOT train dialogue behavior
- Focus on speech generation layers

Target:
- Align text → speaker-specific audio output
- Preserve base conversational reasoning ability

Evaluation:
- speaker similarity
- voice consistency across prompts
- stability under different text domains

Constraints:
- NO dataset augmentation with fake dialogue
- NO mixing with external dialogue datasets
- NO modification of tokenizer or processor

Expected Outcome:
- Model responds conversationally (base CSM ability)
- Output speech matches target speaker (LoRA adaptation)