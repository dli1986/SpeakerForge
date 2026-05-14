# SpeakerForge Pipeline

从 Bilibili 视频提取说话人音频，训练 CosyVoice2 TTS 声音克隆模型的完整流水线。

---

## 目录结构

```
SpeakerForge/
├── pipeline.yaml          # 全局配置（所有 stage 参数）
├── sources.yaml           # 视频源列表（BV 号 + speaker 名）
└── speakerforge/
    ├── raw_sources/       # Stage 0 输出：原始下载文件
    ├── processed/         # Stage 1-6 中间产物
    ├── dataset/           # Stage 7 输出：训练数据集
    ├── cosyvoice_data/    # Stage 8 中间产物：Kaldi 格式数据
    ├── models/            # Stage 8 输出：微调后模型权重
    └── eval/              # Stage 9 输出：推理结果与评估报告
```

---

## 阶段说明

### Stage 0 — 下载视频
**输入**：`sources.yaml`（BV 号列表）  
**输出**：`raw_sources/bilibili/{speaker}/BV*.mp4`  
**行为**：跳过已下载的文件（幂等）  
**命令**：`python __main__.py stage0 --speaker {speaker} --config ../pipeline.yaml`

---

### Stage 1 — 音频提取与重采样
**输入**：`raw_sources/bilibili/{speaker}/BV*.mp4`  
**输出**：`processed/{speaker}/audio/BV*.wav`（24kHz mono）  
**行为**：跳过已处理的文件（幂等）  
**命令**：`python __main__.py stage1 --speaker {speaker} --config ../pipeline.yaml`

---

### Stage 2 — 人声分离（Demucs）
**输入**：`processed/{speaker}/audio/BV*.wav`  
**输出**：`processed/{speaker}/vocals/BV*.wav`（仅人声）  
**行为**：跳过已处理的文件（幂等）  
**命令**：`python __main__.py stage2 --speaker {speaker} --config ../pipeline.yaml`

---

### Stage 3 — 语音分割（VAD）
**输入**：`processed/{speaker}/vocals/BV*.wav`  
**输出**：`processed/{speaker}/segments/BV*_NNNN.wav`（按 silero-vad 切片）  
**行为**：每次重跑覆盖所有输出（非幂等）  
**⚠️ 注意**：手动删除 segments 里的文件是有效的人工清理方式，但重跑 stage3 会重新生成全部切片  
**命令**：`python __main__.py stage3 --speaker {speaker} --config ../pipeline.yaml`

---

### Stage 4 — 语音转录（Whisper）
**输入**：`processed/{speaker}/segments/BV*_NNNN.wav`  
**输出**：`processed/{speaker}/transcripts/BV*_NNNN.json`（含 text、words、confidence）  
**行为**：跳过已有转录的文件（幂等）  
**⚠️ 注意**：手动删除 segments 后，需同步删除对应 transcripts，运行 `python stages/stage3_4_cleanup.py --speaker {speaker}`  
**命令**：`python __main__.py stage4 --speaker {speaker} --config ../pipeline.yaml`

---

### Stage 5 — 质量过滤
**输入**：`processed/{speaker}/segments/*.wav` + `processed/{speaker}/transcripts/*.json`  
**输出**：`processed/{speaker}/filtered.json`（通过时长、置信度、SNR 过滤的样本列表）  
**行为**：每次重跑完全重新生成（幂等且干净）  
**配置**：`pipeline.yaml` stage5 节（min_duration、min_confidence、min_snr_db 等）  
**命令**：`python __main__.py stage5 --speaker {speaker} --config ../pipeline.yaml`

---

### Stage 6 — 音频归一化
**输入**：`processed/{speaker}/segments/*.wav`（filtered.json 中的条目）  
**输出**：`processed/{speaker}/normalized/BV*_NNNN.wav`（RMS 归一化）  
**行为**：每次重跑完全重新生成（覆盖旧文件）  
**命令**：`python __main__.py stage6 --speaker {speaker} --config ../pipeline.yaml`

---

### Stage 7 — 打包数据集
**输入**：`processed/{speaker}/filtered.json` + `processed/{speaker}/normalized/*.wav`  
**输出**：  
- `dataset/{speaker}_versionB/metadata.csv`（filename, text, duration）  
- `dataset/{speaker}_versionB/wavs/*.wav`（归一化音频副本）  
**行为**：每次重跑完全重新生成，**自动清理 wavs/ 中不在 filtered.json 里的旧文件**  
**命令**：`python __main__.py stage7 --speaker {speaker} --config ../pipeline.yaml`

---

### Stage 8 — CosyVoice2 微调（WSL）
**前提**：WSL Ubuntu 中已安装 CosyVoice 环境（见 `pipeline.yaml` stage8 配置）  
**输入**：`dataset/{speaker}_versionB/metadata.csv` + `dataset/{speaker}_versionB/wavs/`  
**中间产物**：`cosyvoice_data/{speaker}_versionB/`（Kaldi 格式、embedding、parquet）  
**输出**：`models/{speaker}_versionB/epoch_*_whole.pt`（每 epoch 保存一次）  
**行为**：每次重跑从 pretrained checkpoint 重新训练（有随机性）  
**配置**：`pipeline.yaml` stage8 节（epochs、cosyvoice2_repo、pretrained_model_dir 等）  
**命令**：`python __main__.py stage8 --speaker {speaker} --config ../pipeline.yaml`

---

### Stage 9 — 推理与评估
**前提**：stage8 已完成，WSL 环境可用  
**输入**：  
- `models/{speaker}_versionB/epoch_*_whole.pt`（微调 checkpoint）  
- `dataset/{speaker}_versionB/wavs/`（参考音频）  
- `pipeline.yaml` stage9 节（checkpoint、ref_wav、test_texts）  
**输出**：  
- `eval/{speaker}_versionB/generated/`（微调模型 zero-shot 推理）  
- `eval/{speaker}_versionB/generated_sft/`（微调模型 SFT 推理）  
- `eval/{speaker}_versionB/generated_baseline/`（原始预训练 zero-shot 对比）  
- `eval/{speaker}_versionB/generated/infer.py`（本次推理的代码快照，可追溯）  
- `eval/{speaker}_versionB/similarity_report.json`（resemblyzer 相似度评分）  
**行为**：每次重跑重新生成 merged_model 并推理  
**配置**：  
```yaml
stage9:
  checkpoint: "epoch_10_whole.pt"  # 留空使用最新
  ref_wav: "processed/{speaker}/segments/BV*.wav"  # 留空自动选第一条
  test_texts:
    - "..."
```
**命令**：`python __main__.py stage9 --speaker {speaker} --config ../pipeline.yaml`

---

## 标准运行流程

### 首次完整运行
```
stage0 → stage1 → stage2 → stage3 → stage4 → stage5 → stage6 → stage7 → stage8 → stage9
```

### 新增视频源后
```
stage0 → stage1 → stage2 → stage3 → stage4 → stage5 → stage6 → stage7 → stage8 → stage9
```
（stage0-4 会跳过已处理的文件，stage5 起完全重新生成）

### 人工清理 segments 后
```
# 1. 手动删除 processed/{speaker}/segments/ 中不合适的 wav
# 2. 差分清理对应 transcripts
python stages/stage3_4_cleanup.py --speaker {speaker}
stage5 → stage6 → stage7 → stage8 → stage9
```

### 调整过滤参数（pipeline.yaml stage5）后
```
stage5 → stage6 → stage7 → stage8 → stage9
```

### 只换 checkpoint 测试推理
```
# 修改 pipeline.yaml stage9.checkpoint
stage9
```

---

## 各阶段依赖关系

```
stage0
  └─ stage1
       └─ stage2
            └─ stage3  ←── 人工清理 segments 在此层
                 └─ stage4  ←── 差分清理 transcripts 与 stage3 对齐
                      └─ stage5  ←── filtered.json（每次从头生成）
                           └─ stage6
                                └─ stage7  ←── dataset/ 每次干净重建
                                     └─ stage8  ←── 有随机性，保存所有 epoch checkpoint
                                          └─ stage9
```

---

## 重要注意事项

1. **手动清理 segments 后，必须同步清理 transcripts**，否则 stage5 会因找不到 transcript 而跳过对应 wav，导致实际过滤结果不符合预期
2. **stage8 训练有随机性**，相同数据不同轮次结果会有差异。通过 stage9 的 `checkpoint` 配置可以评估不同 epoch 的效果
3. **stage9 的 ref_wav 和 ref_text 必须匹配**，stage9 会从 metadata.csv 按文件名查找对应文本，确保两者一致
4. **不要手动修改 dataset/wavs/ 或 metadata.csv**，这两个文件由 stage7 完全管理
