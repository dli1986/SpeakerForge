SYSTEM CONTEXT:
We are building a conversational speech system based on CSM-1B.

We have already completed:
- Speaker dataset pipeline (audio + text)
- Speaker validation via CosyVoice (SFT success, similarity ~0.8+)

We are now in Phase 2:
Inject the target speaker (UP主) voice into CSM using LoRA.

---

OBJECTIVE:

Train a LoRA adapter on CSM-1B to inject a single speaker’s voice and speaking style,
WITHOUT modifying or degrading the model’s intrinsic conversational ability.

This is voice adaptation, NOT dialogue training.

---

DATA:

Use the existing dataset:
- Single speaker (Akinokoe)
- Format: (audio, text)
- ~600–1000 clean samples
- No conversation structure
- No multi-speaker data

Expected HF dataset format:
{
  "audio": <audio_path_or_array>,
  "text": <string>
}

---

STRICT CONSTRAINTS (VERY IMPORTANT):

DO NOT:
- Add or construct conversation data
- Use dialogue datasets (e.g. dailytalk)
- Create fake Q&A or multi-turn samples
- Perform full fine-tuning
- Modify tokenizer or processor
- Change model architecture

We are NOT training dialogue behavior.

---

TRAINING STRATEGY:

- Use LoRA only (parameter-efficient tuning)
- Freeze base CSM-1B weights
- Apply LoRA to:
  → transformer layers (attention / MLP)
  → optionally audio projection layers

- Training objective:
  text → audio generation alignment

- This is equivalent to TTS-style adaptation inside CSM.

---

EXPECTED OUTPUT:

After training:
- The model should generate speech in the target speaker’s voice
- The model should still preserve its original conversational ability
- Voice should remain consistent across different text prompts

---

EVALUATION:

Check:
- speaker similarity (resemblyzer)
- subjective listening (naturalness)
- consistency across:
  → AI text
  → casual text
  → long text

DO NOT evaluate dialogue ability here.

---

OUT OF SCOPE (Phase 3):

The following is NOT part of this phase:
- Conversation loop
- Memory / session handling
- Multi-turn audio chaining
- Real-time inference system

These will be implemented later.

---

FUTURE PIPELINE (DO NOT IMPLEMENT NOW):

Phase 3 will introduce:
- conversation history accumulation
- audio + text context loop
- streaming generation