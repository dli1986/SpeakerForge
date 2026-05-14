# SpeakerForge Phase 2 — CSM-1B LoRA 声音注入

基于 Phase 1 构建的音频数据集，通过 Unsloth LoRA 微调 Sesame CSM-1B，将目标说话人的声音注入模型。

---

## 目录结构

```
SpeakerForge/sesame/
├── config.yaml            # 全局配置（所有 stage 参数）
├── config.py              # SesameConfig 数据类 + load_sesame_config()
├── cli.py                 # Click CLI 入口（stage1 / stage2 / stage3）
├── __main__.py            # python -m sesame 入口
├── requirements.txt       # 依赖列表
├── colab_train.ipynb      # ★ Colab T4 训练 notebook（stage2 推荐入口）
├── stages/
│   ├── stage1_prep.py     # 数据准备（本地运行）
│   ├── stage2_train.py    # LoRA 训练（本地 GPU 可用时）
│   └── stage3_eval.py     # 推理 + 相似度评估（本地运行）
├── data/                  # Stage 1 输出：HF 格式数据集（可 gitignore）
├── models/                # Stage 2 输出：LoRA adapter 权重（可 gitignore）
└── eval/                  # Stage 3 输出：生成音频 + 评估报告
```

**Phase 1 路径（只读，绝不写入）：**
```
SpeakerForge/speakerforge/dataset/Akinokoe_versionB/   ← stage1/stage3 读取此路径
    ├── metadata.csv
    └── wavs/*.wav
```

---

## 隔离保证

> **Phase 2 的三个 stage 均不会修改 Phase 1 的任何文件。**

| Phase 1 路径 | Phase 2 访问方式 |
|---|---|
| `../speakerforge/dataset/*/metadata.csv` | 只读（stage1 读取文本和文件名） |
| `../speakerforge/dataset/*/wavs/*.wav` | 只读（stage3 读取参考音频用于相似度计算） |
| `../speakerforge/processed/` | 不访问 |
| `../speakerforge/models/` | 不访问 |
| `../speakerforge/eval/` | 不访问 |

Phase 2 的所有输出均写入 `sesame/data/`、`sesame/models/`、`sesame/eval/`（相对于执行目录 `SpeakerForge/sesame/`）。

---

## 阶段说明

### Stage 1 — 数据准备（本地运行）
**输入**：`../dataset/{speaker}_version{version}/metadata.csv` + `wavs/*.wav`（Phase 1 stage7 输出）  
**输出**：`data/{speaker}_v{version}/`（HuggingFace Dataset 格式，存储到磁盘）  
**行为**：每次重跑清空并重建输出目录  
**命令**：
```bash
cd SpeakerForge/sesame
python -m sesame stage1 --config config.yaml
```

---

### Stage 2 — LoRA 训练

**推荐入口：`colab_train.ipynb`（Colab T4）**  
**备用入口：`python -m sesame stage2`（本地 GPU ≥ 8GB）**

**输入**：`data/{speaker}_v{version}/`（stage1 输出的 HF 数据集）  
**输出**：`models/{speaker}/lora_adapter/`（LoRA adapter + processor 权重）  
**行为**：每次重跑从预训练权重重新训练（有随机性）

#### Colab 工作流（利用已挂载的 G: 盘）

```
① 本地 stage1 完成后：
   sesame/data/Akinokoe_vB/  →  复制到  G:\SpeakerForge\sesame\data\Akinokoe_vB\

② 打开 colab_train.ipynb，上传到 Colab 或直接从 Drive 打开

③ Cell 2：修改 DRIVE_ROOT 路径后挂载 Drive
   DRIVE_ROOT = "/content/drive/MyDrive/SpeakerForge/sesame"

④ 逐 cell 执行（Cell 1 安装 → Cell 3 加载模型 → Cell 4 预处理 → Cell 5 训练）

⑤ Cell 6 执行后 adapter 保存到 Drive，本地 G: 盘同步可见：
   G:\SpeakerForge\sesame\models\Akinokoe\lora_adapter\

⑥ 本地运行 stage3 进行评估（直接读取 G: 路径，或先复制到本地 models/）
```

#### 训练参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| LoRA rank (r) | 32 | 官方推荐值 |
| target_modules | 7个（全部 proj 层） | q/k/v/o/gate/up/down |
| max_steps | 120 | 官方默认 60；1492 samples 建议 120-200 |
| batch_size | 2 × grad_accum 4 | 等效 batch 8 |
| Trainer | `transformers.Trainer` | CSM 多模态输入，不能用 SFTTrainer |
| load_in_4bit | **false** | 音频质量要求，不可改为 true |

---

### Stage 3 — 推理与评估（本地运行）
**前提**：`models/{speaker}/lora_adapter/` 存在（可来自 Colab/Drive 或本地训练）  
**输入**：
- `models/{speaker}/lora_adapter/`（LoRA adapter）
- `../dataset/{speaker}_version{version}/wavs/`（参考音频，仅读取）
- `config.yaml` 中 `eval.test_texts`

**输出**：
- `eval/{speaker}/generated/gen_NNNN.wav`（生成音频）
- `eval/{speaker}/similarity_report.json`（resemblyzer 余弦相似度报告）

**行为**：每次重跑清空 `eval/{speaker}/generated/` 并重新生成  
**命令**：
```bash
cd SpeakerForge/sesame
python -m sesame stage3 --config config.yaml
```

---

## 标准运行流程

### 首次完整运行（Colab T4）
```
[本地] stage1
  → 复制 data/ 到 G:\SpeakerForge\sesame\data\
  → [Colab] colab_train.ipynb（stage2）
  → adapter 自动同步到 G: 盘
  → 复制/移动 adapter 到 sesame/models/Akinokoe/lora_adapter/（可选）
  → [本地] stage3
```

### 修改训练参数后重新训练
```
# 修改 config.yaml 中 lora / training 节（或直接修改 colab_train.ipynb 中的变量）
[Colab] colab_train.ipynb → [本地] stage3
```
（stage1 输出不变，无需重跑）

### 只换测试文本重新评估
```
# 修改 config.yaml 中 eval.test_texts
[本地] stage3
```

### Phase 1 数据集更新后（重跑 phase1 stage7）
```
[本地] stage1 → 复制到 Drive → [Colab] stage2 → [本地] stage3
```

---

## 各阶段依赖关系

```
[Phase 1 stage7 输出]
  ../speakerforge/dataset/{speaker}_versionB/
       │
       ▼
  stage1 (本地)              ← metadata.csv + wavs → HF Dataset (data/)
       │
       │  复制 data/ → Google Drive (G: 盘)
       ▼
  colab_train.ipynb (Colab)  ← 读取 Drive data/ → 训练 → 保存 adapter 到 Drive
       │
       │  adapter 通过 G: 盘同步到本地 models/
       ▼
  stage3 (本地)              ← 加载 adapter + 参考音频 → 生成 + 相似度评估
```

---

## config.yaml 关键参数说明

```yaml
dataset:
  root: "../speakerforge/dataset"  # Phase 1 dataset 路径，相对于 sesame/ 目录
  speaker: "Akinokoe"
  version: "B"

model:
  name: "unsloth/csm-1b"
  load_in_4bit: false      # 必须 false，音频质量要求

lora:
  r: 32
  alpha: 32
  target_modules:          # 全部 7 个 projection 层（官方标准配置）
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  max_steps: 120           # 官方默认 60；本项目数据量大，建议 120-200
  output_dir: "models/Akinokoe"

eval:
  max_new_tokens: 500      # ~40s 音频上限；125 tokens ≈ 10s
  n_ref_wavs: 20
  test_texts: [...]
```

---

## 重要注意事项

1. **所有本地命令必须在 `SpeakerForge/sesame/` 目录下运行**，确保相对路径正确
2. **`load_in_4bit` 必须为 `false`**，这是 CSM 音频质量的硬性要求
3. **Colab 路径修改**：打开 `colab_train.ipynb` 后先修改 Cell 2 中的 `DRIVE_ROOT`，确保指向你 Drive 中实际放置数据的路径
4. **adapter 路径对齐**：从 Drive 取回 adapter 后，确保放置在 `sesame/models/Akinokoe/lora_adapter/`，与 `config.yaml` 中 `output_dir` 一致
5. **相似度基准**：Phase 1（CosyVoice2 SFT）零样本相似度约 0.84；CSM 架构不同，基准不同，初次运行以 ≥ 0.65 为合格线
6. **训练步数**：`max_steps=120` 时 T4 约 20-25 分钟；可在 Cell 5 中直接调整 `MAX_STEPS`
