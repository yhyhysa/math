# STATUS — E题工程开发

- **接收状态**：已接收 T0 复验返工（DeepSeek Harness 执行，等待 Codex 独立复验）
- **当前阶段**：**T0 复验返工完成，未签署 M1 PASS，未启动 T1**
- **本文件生成方式**：由 `src.audit` 于 **2026-09-23 17:14:53** 依据**本次运行**的返回值、耗时、退出码与产物哈希动态生成（不再写死历史数值）
- **执行主体**：DeepSeek Harness（`deepseek-flash`）
- **只读输入**：`E题数据.zip`（SHA-256 `F895003ABEB3E972E69EF9C457032720CBCB8ECD3D2261A0E30BC15667604ACB`）、题面 DOCX、`训练demo/`、`视频demo/`
- **Python**：`D:/第二十三届中国研究生数学建模竞赛 - 中文题目/中文题目/E题/训练demo/.venv/Scripts/python.exe`
- **环境**：Windows-11-10.0.26200-SP0；GPU `NVIDIA GeForce RTX 5070 Ti`（1 卡），CUDA 12.8，torch 2.7.1+cu128，transformers 5.17.0
- **备份**：返工前产物与源码备份于 `results/t0_before_fix/`（`round2_pre_*`、`round2_pre_src/`）

## 一、本轮主入口实测运行

- 命令：`& "D:/第二十三届中国研究生数学建模竞赛 - 中文题目/中文题目/E题/训练demo/.venv/Scripts/python.exe" -m src.audit --config configs/base.json`
- 结论：**全部子检查 PASS**（见下表）；总耗时 93.36s
- 失败语义：任一子检查失败即**非零退出**，且**不写 PASS、不覆盖既有 STATUS**；失败注入实测记录见 `results/t0_before_fix/t0_round2_run_receipt.json` 与 `results/t0_before_fix/fail_injection_evidence/`（在隔离副本中注入，未改动任何原始数据）。

| # | 步骤 | 状态 | 退出码 | 耗时(s) | 产物 | 实测摘要 |
|---|---|---|---|---|---|---|
| 1 | `integrity_gate_pre` | PASS | 0 | 14.9 | `D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\工程\results\codex_review\raw_integrity_20260923T091320384942Z.json` | PASS: checked=245, errors=[], report=D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\工程\results\codex_review\raw_integrity_20260923T091320384942Z.json |
| 2 | `environment` | PASS | 0 | 0.0 | - | GPU=NVIDIA GeForce RTX 5070 Ti, CUDA=12.8, torch=2.7.1+cu128 |
| 3 | `unified_reader` | PASS | 0 | 0.26 | `results/reader_check.json` | A2 train/valid/test=[3395, 728, 727], A3 files=30, A4 files=20 |
| 4 | `input_manifest` | PASS | 0 | 4.07 | `results/input_manifest.csv` | rows=245 |
| 5 | `cross_attachment_overlap` | PASS | 0 | 0.24 | `results/cross_attachment_overlap.csv` | sample_ids=18 {'train': 11, 'valid': 0, 'test': 7}, video_ids=24 |
| 6 | `overlap_manifest_verify` | PASS | 0 | 0.73 | `results/overlap_manifest_verify.json` | independent checker exit=0 in 0.73s |
| 7 | `label_consistency` | PASS | 0 | 1.11 | `results/label_consistency_audit.json` | 18/18 overlapping samples: regression labels identical, transcripts identical (1.11s) |
| 8 | `bert_full_audit` | PASS | 0 | 8.16 | `results/bert_audit_full.json` | revision=86b5e0934494, evaluated=4870, max_err=5.966e-04, >1e-3=0 |
| 9 | `mask_audit` | PASS | 0 | 0.89 | `results/mask_audit.json` | mask/dtype statistics written in 0.89s |
| 10 | `video_detailed_audit` | PASS | 0 | 47.94 | `results/video_audit_detailed.json` | container_discrepancy=90/100, cv==pyav=100/100 |
| 11 | `integrity_gate_post` | PASS | 0 | 14.86 | `D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\工程\results\codex_review\raw_integrity_20260923T091438831329Z.json` | PASS: checked=245, errors=[], report=D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\工程\results\codex_review\raw_integrity_20260923T091438831329Z.json |

## 二、分项实测结论

### 1. 统一读取器（附件二 / 附件三 / 附件四，51 个真实 pkl 文件）

- 附件二 `aligned_50.pkl`（列式 split 字典）：train 3395、valid 728、test 727 条，ID 唯一数分别为 3395/728/727，形状错误 0 条。
- 附件三对齐版：30 个文件 → 30 条样本（去 batch=1），`sample_id` 取文件名 stem，未制造官方 ID 或 `text` 向量；失败文件 0。
- 附件四对齐版：20 个文件 → 20 条样本，保留原始 `id`/`raw_text`；失败文件 0。
- 产物 `results/reader_check.json`（`all_passed=True`）。

### 2. 附件一 `label-100.xlsx` 与跨附件交集（重算）

- `label` 工作表：100 行，极性分布 {'Positive': 57, 'Neutral': 25, 'Negative': 18}；`label<0/==0/>0` 与 Negative/Neutral/Positive 逐行一致（mismatch 0）。
- 精确键 `video_id + '$_$' + clip_id` 与附件二交集：**sample_id 18 个**（train 11、valid 0、test 7）；**源 video_id 24 个**（train 16、valid 2、test 6）。
- 附件二 train/valid/test 之间 video_id 交集：{'train_valid': 0, 'train_test': 0, 'valid_test': 0}（全 0）。
- **说话人独立性未知**：数据集无说话人 ID 字段，仅能说明源 video_id 跨 split 不重叠。
- 独立复核器 `src.tools.verify_cross_overlap`（从原始 xlsx/pkl 独立重算并与交付 CSV 比对）退出码 0，证据 `results/overlap_manifest_verify.json`。
- 补充实测：18 条重叠样本在附件一与附件二中的**回归标签与转写文本完全一致（18/18）**，证据 `results/label_consistency_audit.json`；Q2/Q3 参数仍只从附件二 train 学习，未把附件一并入训练。

### 3. 输入文件清单（245 行）

- 解压目录实测文件 244 个（含 `.DS_Store` 1 个）；清单登记 244 个数据文件 + 原始 zip 1 行 = **245 行**，无重复、哈希非空。
- 独立复核器同时校验清单路径集合与磁盘文件集合严格相等（`manifest_path_set_matches_disk=True`）并抽检重算 SHA-256。
- `.DS_Store` 不作为建模输入；未对齐版 30+20 个 pkl 与 20 个 mp4 标 `not_used_in_primary_pipeline` 并写明排除理由。

### 4. 文本编码全量核验（版本绑定）

- 编码器：`bert-base-uncased`，**固定 revision `86b5e0934494bd15c9632b12f734a8a67f723594`**，从快照目录 `C:\Users\本地账户\.cache\huggingface\hub\models--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594` 加载（`local_files_only=True`，`HF_HUB_OFFLINE=1`），实测加载路径与该快照一致：`load_binding_verified=True`；加载前后文件哈希均核对。
- `vocab.txt` `07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3`（校验 True）；`config.json` `7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20`；`model.safetensors` `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`；`tokenizer.json` `ce64fce797c24f68df90b40a3f74f579b336a493db14bd583fd520ea0d8c9a98`。
- 重分词核对（`input_ids`/`attention_mask`/`token_type_ids` 三行同时相等）：**4870/4870**（附件二 train 3395/3395、valid 728/728、test 727/727；附件四 20/20）。
- 冻结编码器（`eval()` + `no_grad()`）从 `text_bert` 重算 `last_hidden_state` 与现有 `text` 比较，实际评估样本数 **4870**，阈值 0.001：

| 数据组 | 样本数 | 全张量最大绝对误差 | 均值 | p50 | p99 | 超阈值数 | batch | 耗时(s) |
|---|---|---|---|---|---|---|---|---|
| 附件二 train | 3395 | 5.966e-04 | 1.215e-05 | 7.153e-06 | 1.046e-04 | 0 | 64 | 4.72 |
| 附件二 valid | 728 | 3.104e-04 | 1.279e-05 | 7.391e-06 | 8.975e-05 | 0 | 64 | 0.95 |
| 附件二 test | 727 | 3.276e-04 | 1.130e-05 | 6.914e-06 | 7.263e-05 | 0 | 64 | 0.9 |
| 附件四 对齐版 | 20 | 9.561e-05 | 1.701e-05 | 1.132e-05 | 8.188e-05 | 0 | 20 | - |

- 全量最大绝对误差 5.966e-04、均值 1.214e-05、p99 9.794e-05；超 0.001 样本 **0**；max error ≤ 1e-4 的样本 4821/4870。
- 误差原因写法：**推断（INFERRED）**——The residual differences are *inferred* to come from FP32 GPU (cuDNN/CUDA) matrix-multiplication accumulation order differing from the environment that produced the official 768-dim vectors. This is an inference from the error magnitude and distribution, not a directly measured root cause; the official extraction environment is not available for comparison.
- 附件三（30 文件）：**没有 raw_text，因此不能声称重分词已验证**；仅核验 CLS=101（30/30）、SEP=102（30/30）、PAD=0（30/30）、token 范围 [0, 28428] 与冻结编码器可运行（输出 [30, 50, 768]，passed=True）。
- **不宣称**“全部完全无损”：本结论覆盖的是被实际评估的 4870 条样本的重构误差分布。

### 5. 视频与掩码结论（限证据范围）

- 附件一 100 条视频：容器标称帧数与实际解码帧数不符 **90/100**；OpenCV 与 PyAV 解码帧数一致 **100/100**；记录到的 PTS 严格单调 100/100；带有效 PTS 帧数与解码帧数相等的样本 100/100（缺 PTS 帧的样本 0 条）。
- 帧数差异根因：**待确认**（未检查 MP4 atom 与 H.264 NAL 码流，不能写成“切片前元数据残留”已证明）。
- 音频：检出音轨并可解码的样本 100/100；**不因存在音轨即宣称“音轨完整”**。解码时长与容器时长偏差 > 5% 的样本 2 条：
   - `-9y-fZ3swSY$_4`: 容器 2.821s，音频解码 2.6648s，偏差 5.54%
   - `-s9qJ7ATP7w$_6`: 容器 2.475s，音频解码 2.3148s，偏差 6.47%
- 时间映射：未取得真实词级时间映射，Q1/Q3 的词时间戳标 `unresolved`，不得由 50 个位置等分推断。
- 掩码/全零：padding 位置、有效 token 位置内的全零音视频特征、以及训练时模拟缺失**分列统计**（下表），完整数据见 `results/mask_audit.json`；有效位置内全零标 **“疑似无观测”**，缺乏原始提取证据，不宣称人脸遮挡或真实模态缺失。

| 划分 | 样本 | padding 位 | 有效位 | audio dtype | vision dtype | audio 有效位全零(去CLS/SEP) | vision 有效位全零(去CLS/SEP) | 含此类全零的样本(audio/vision) |
|---|---|---|---|---|---|---|---|---|
| train | 3395 | 86078 | 83672 | float64 | float64 | 65 | 4249 | 4/421 |
| valid | 728 | 17772 | 18628 | float64 | float64 | 46 | 980 | 1/85 |
| test | 727 | 18041 | 18309 | float64 | float64 | 0 | 1024 | 0/99 |

- 解释边界：有效位置内全零一律记为 **疑似无观测**，缺少原始特征提取证据时不宣称为人脸遮挡或真实模态缺失。
- 训练时模拟缺失：**T1 待实现**，T0 未执行（`executed_in_t0=false`）。

- dtype：附件二 `audio`/`vision` 原始为 **float64**（实测，见上表）；训练时在 Dataset/DataLoader 处转 FP32 并做有限值检查——**该实现属 T1，T0 未执行**。
- 74 维语音特征逐维语义 **未知**，不猜测命名。

## 三、主要产物 SHA-256（本次运行实测）

- `data/raw/E题数据.zip`: `F895003ABEB3E972E69EF9C457032720CBCB8ECD3D2261A0E30BC15667604ACB`
- `results/interface_contract.json`: `5d4d945a2934b98a9c4b914be800b1c24b73c65418a82399cfd6feccb5297241`
- `results/reader_check.json`: `2179381c1837f811f4f3020fce8b10f0f577fd3ebb9fe0791197cc77cbdeff64`
- `results/input_manifest.csv`: `c233346e7963b3a57c6edd2ce6641f6ef83945e48d76886e7da2c856e3fa2896`
- `results/cross_attachment_overlap.csv`: `7a12bdb265ab87ac1321b59b67803044b2a83084420ab5046b64e6dd3699d451`
- `results/overlap_manifest_verify.json`: `2ff50ddca1b9b99bfb93209f2d00fa8c6caef975b3a0365fa6273bcd94397150`
- `results/label_consistency_audit.json`: `69f52e31f4463d5f5fc1e1a49abcaacb4e2f2310349bec1ea9d887b49626557a`
- `results/bert_audit_full.json`: `b023ed650099151aef8a2a5aaaebc185a24c577882aba47092a0b883d0e0c18b`
- `results/video_audit_detailed.json`: `623d29b160cf1ad193c05175c06417aaa53ad4659bee42ef2d13002a5b13f8a5`
- `results/mask_audit.json`: `885ae31a95e37e1db1fda9f29f65019ae5fc9098a06405e856ac15485c393914`
- `results/STATUS.md`: `（见 results/artifacts_sha256.json）`

## 四、未决事项 / 不做结论的部分

1. 附件一容器标称帧数差异的根因未证实（需 atom/NAL 解析），当前仅记录实测现象。
2. 音频流“完整/无截断”未证实；2 条 >5% 偏差样本仅记录实测值。
3. 说话人独立性未知（无说话人字段）。
4. 74/35 维特征逐维语义未知。
5. 词级真实时间映射未取得 → Q1/Q3 相关位置标 `unresolved`。
6. 50MB 提交包预算为**有依据的估算**（算法与依据见 `results/data_audit.json` 的 `submission_package_budget`），非实测打包结果。
7. 冻结 BERT 权重（~440MB）不进提交包；外部下载是否被官方规则接受**待官方规则核验**。
8. T1 待实现：Dataset/DataLoader 的 dtype 转换与 `torch.isfinite` 逐批检查、训练时缺失掩码模拟。

## 五、当前状态

- **T0 复验返工完成，等待 Codex 独立复验**；**未签署 M1 PASS**，**未启动 T1**。
