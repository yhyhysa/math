# E题 T0 复验返工回执（DeepSeek Harness → Codex 独立复验）

- 日期：2026-09-23
- 范围：**仅 T0 数据与环境预检**；未启动 T1，未训练模型，未改动任何原始附件与 demo。
- 结论：**本轮五项返工已全部实测完成；所有检查真实通过。未自行签发 M1 PASS，等待 Codex 独立复验。**
- 执行者：DeepSeek Harness（模型 `deepseek-flash`），接替原文档中的 Antigravity 执行者角色。
- 原始 zip SHA-256：`F895003ABEB3E972E69EF9C457032720CBCB8ECD3D2261A0E30BC15667604ACB`（本轮多次复核一致）。

## 1. 修改文件清单

| 文件 | 改动要点 |
|---|---|
| `src/audit.py` | 主入口重构：任一子检查失败→**非零退出**且**不写 PASS/不覆盖既有 STATUS**；完整性检查作为 **Gate 1（前置）/Gate 2（后置）** 两个步骤并检查退出码；STATUS 与主审计输出**按本次运行**动态生成（误差/命令/耗时/退出码/SHA-256/未决项）；新增 `StepResult`、`run_step`、`run_integrity_gate`、`run_cli` 崩溃兜底（非零退出 9）；子进程强制 UTF-8 I/O；新增 `--skip-integrity`（测试专用）与 `DSH_T0_FAIL_INJECTION` 失败注入钩子（正常运行时无副作用）；接口合同的 dtype/掩码/T1 待实现措辞收紧 |
| `src/tools/audit_bert_full.py` | 按固定 revision `86b5e0934494bd15c9632b12f734a8a67f723594` 的**本地快照目录**加载，`local_files_only=True` + 离线标志；运行前核对 vocab/config/权重/tokenizer 哈希，加载后复检，**缺失或哈希不符即非零退出**；实测加载路径须等于报告路径；新增 `--self-check`；`reconstruction_fidelity` 计数全部由数组计算、差异原因标注为**推断** |
| `src/tools/safe_extract.py` | 已有文件**只读**：按内容（CRC32 + SHA-256）比较，一致跳过，不一致**报错退出 2 且绝不覆盖**；不再以“大小相同”判定；写入后复算哈希；所有写入改为 `open(..., "xb")` |
| `src/tools/audit_videos_detailed.py` | 逐条新增：有效 PTS 帧数、缺 PTS 帧数、PTS 覆盖比、PTS 跨度、音频覆盖比、音轨完整性“无法判定”、时间映射 `unresolved`、根因“待确认”；汇总新增 PTS 覆盖统计 |
| `src/tools/verify_cross_overlap.py`（新增） | 从原始 xlsx/pkl **独立重算**交集与清单完整性，并与交付 CSV 比对；不一致非零退出 |
| `src/tools/audit_label_consistency.py`（新增） | 18 条重叠样本的附件一/二标签与转写一致性核对 |
| `src/tools/mask_audit.py`（新增） | padding / 有效位全零 / dtype 分列统计（含 CLS/SEP 剔除后的统计） |
| `src/tools/run_audit_testroot.py`（新增） | **测试专用** shim：把审计指向 `src/`+`configs/` 副本，读真实只读输入、写副本输出；`DSH_T0_TESTROOT` 未设置时行为与正常运行完全一致 |
| `README.md` | 可复制的绝对 Python 路径命令、新工具、失败语义、编码器复现边界、结论边界 |
| `STATUS.md` | 由 `src.audit` 动态生成（见下） |
| `results/*.json`、`results/input_manifest.csv` | 本轮重新生成 |

保留的历史快照：`results/t0_before_fix/round2_pre_*`（返工前产物）、`results/t0_before_fix/round2_pre_src/`（返工前源码）。**未删除或覆盖任何原始数据与校验基准。**

## 2. 真实命令、退出码与耗时（全部实跑）

| # | 命令（工作目录 `E题\工程`） | 退出码 | 耗时 | 主要产物 |
|---|---|---|---|---|
| 1 | `& "<venv>\python.exe" -m src.tools.verify_raw_integrity`（改动前） | **0** | 15.46 s | `results/codex_review/raw_integrity_*.json` |
| 2 | `python -m src.check_data_reader` | **0** | 0.4 s | `results/reader_check.json` |
| 3 | `python -m src.tools.cross_overlap` | **0** | 0.75 s | `results/cross_attachment_overlap.csv` |
| 4 | `python -m src.tools.verify_cross_overlap` | **0** | 0.75 s | `results/overlap_manifest_verify.json` |
| 5 | `python -m src.tools.audit_label_consistency` | **0** | 1.14 s | `results/label_consistency_audit.json` |
| 6 | `python -m src.tools.mask_audit` | **0** | 0.9 s | `results/mask_audit.json` |
| 7 | `python -m src.tools.audit_bert_full --self-check` | **0** | 4.16 s | 快照哈希核对 |
| 8 | `python -m src.tools.safe_extract --zip data/raw/E题数据.zip --out <临时目录> --only <附件4对齐版>`（三次） | **0 / 0 / 2** | 1.4 / 1.1 / 1.0 s | 首解压→同内容跳过→**篡改后冲突退出 2 且未覆盖** |
| 9 | `python -m src.audit --config configs/base.json`（最终权威运行） | **0** | **97.6 s**（内部汇总 92.9 s） | `data_audit.json`、`interface_contract.json`、`STATUS.md`、`audit_run_report.json` |
| 10 | `python -m src.tools.verify_raw_integrity`（改动后 / Gate 2） | **0** | 14.8 s | 同上目录新时间戳 JSON |

主入口内部步骤（同一进程，逐项退出码记录于 `results/audit_run_report.json`）：
`integrity_gate_pre=PASS(0)` → `unified_reader=PASS(0)` → `input_manifest=PASS(0)` → `cross_attachment_overlap=PASS(0)` → `overlap_manifest_verify=PASS(0)` → `label_consistency=PASS(0)` → `bert_full_audit=PASS(0)` → `mask_audit=PASS(0)` → `video_detailed_audit=PASS(0)` → `integrity_gate_post=PASS(0)`。

## 3. 本轮实际复核到的关键数值

- **读取器（51 个真实 pkl）**：附件二 train 3395 / valid 728 / test 727，字段形状 `text(50,768)`、`text_bert(3,50)`、`audio(50,74)`、`vision(50,35)`；附件三 30 文件 → 30 条单样本（去 batch=1，ID 取文件名 stem，未制造官方 ID 或 text）；附件四 20 文件 → 20 条，保留原始 ID/文本。全部 ID 唯一、零形状错误。
- **附件一 `label-100.xlsx`**：100 行，Negative 18 / Neutral 25 / Positive 57；`label<0/==0/>0` 与三类逐行一致（不符 0 行）；100 个 `(video_id, clip_id)` 与 100 个视频文件一一对应（缺文件 0）。
- **交集（精确键 `video_id + '$_$' + clip_id`）**：sample_id **18 个**（train 11、valid 0、test 7）；源 video_id **24 个**（train 16、valid 2、test 6）；附件二三划分之间 video_id 交集 **0**。独立复核器从原始文件重算并与交付 CSV 完全一致；STATUS/data_audit 中原“零交集/严格说话人独立”表述已删除，改为**“说话人独立性未知”**。
- **补充实测（新增证据）**：18 条重叠样本在附件一与附件二中的**回归标签与转写文本完全一致（18/18）**；Q2/Q3 参数仍只从附件二 train 学习，附件一未并入训练。
- **输入清单**：实测解压文件 245 个（含 `.DS_Store` 1 个）；清单登记 244 个数据文件 + 原始 zip 1 行 = **245 行**，无重复、哈希非空；独立复核器验证清单路径集合与磁盘集合严格相等，并抽检 12 个文件重算 SHA-256（差异 0）。
- **文本编码（版本绑定）**：`bert-base-uncased`，固定 revision `86b5e0934494bd15c9632b12f734a8a67f723594`；实际加载路径 = 报告路径（`load_binding_verified=True`）；`vocab.txt` `07eced37…`、`config.json` `7160e155…`、`model.safetensors` `68d45e23…`、`tokenizer.json` `ce64fce7…`。
  - 重分词三行全等：附件二 4850/4850、附件四 20/20（合计 **4870/4870**）。
  - 冻结编码器重构（`eval()`+`no_grad()`，batch=64，RTX 5070 Ti）：**实际评估 4870 条**；整体最大绝对误差 `5.9664e-04`，超 `1e-3` 样本 **0**；train 3395 max `5.97e-04`、valid 728 max `3.10e-04`、test 727 max `3.28e-04`、附件四 20 max `9.56e-05`。
  - 差异原因写为**推断**（FP32/cuDNN 累加顺序），不写成已证实的唯一原因；附件三**无 raw_text，明确不声称重分词验证**，只核验 CLS/SEP/PAD、token 范围与冻结编码器可运行。
- **视频/掩码**：附件一 100 条中容器标称帧数与解码帧数不符 **90/100**；OpenCV 与 PyAV 解码帧数一致 **100/100**；PTS 单调 100/100；**带有效 PTS 帧数 23241 = 解码帧数 23241**（缺 PTS 样本 0）。音频可解码 100/100；偏差 >5% 的 **2 条**：`-9y-fZ3swSY$_4`（容器 2.821 s / 音频 2.6648 s，5.54%）、`-s9qJ7ATP7w$_6`（容器 2.475 s / 音频 2.3148 s，6.47%）。
  - 帧数差异根因：**待确认**（未做 MP4 atom / H.264 NAL 解析）；**不宣称“音轨完整”**；时间映射标 `unresolved`。
  - 掩码/dtype 实测：`audio`/`vision` 原始 **float64**；有效位（剔除 CLS/SEP）全零向量 train 65/4249、valid 46/980、test 0/1024（audio/vision），一律标**疑似无观测**；训练时缺失模拟标 **T1 待实现**。

## 4. 失败注入结果（未改动任何原始数据）

- 方式：`DSH_T0_FAIL_INJECTION=unified_reader` 在 `src/`+`configs/` 副本中运行，读取真实只读输入、写入副本目录（shim `src.tools.run_audit_testroot`）。
- 实测：退出码 **1**；`[FAIL] unified_reader` → `ABORT (exit 1): 子检查失败：unified_reader -> AssertionError: INJECTED FAILURE ...`；副本 `audit_run_report.json` `status=FAIL`；副本 `STATUS.md` 标 **BLOCKED**。
- 关键：真实 `STATUS.md` 与 `results/interface_contract.json` 的 SHA-256 在注入前后**完全未变**（`INJECTION_STATUS_UNCHANGED=True`、`INJECTION_CONTRACT_UNCHANGED=True`），即失败运行**没有覆盖此前通过的结论**。
- 另两项独立失败注入：编码器快照**缺文件**、**vocab 哈希不符**，均退出码 **1**（`--self-check --hub-dir <临时目录>`）。
- 证据：`results/t0_before_fix/fail_injection_evidence/`（副本 STATUS、run report、reader_check、生效配置）、`results/t0_before_fix/final_verification_stdout.txt`、`results/t0_before_fix/t0_round2_run_receipt.json`（含 4 条命令的退出码/耗时/哈希与 `ALL_EXPECTED=true`）。

## 5. 主要产物 SHA-256（权威运行记录值）

| 产物 | SHA-256 |
|---|---|
| `data/raw/E题数据.zip` | `F895003ABEB3E972E69EF9C457032720CBCB8ECD3D2261A0E30BC15667604ACB` |
| `results/input_manifest.csv` | `c233346e7963b3a57c6edd2ce6641f6ef83945e48d76886e7da2c856e3fa2896` |
| `results/cross_attachment_overlap.csv` | `7a12bdb265ab87ac1321b59b67803044b2a83084420ab5046b64e6dd3699d451` |
| `results/interface_contract.json` | `5d4d945a2934b98a9c4b914be800b1c24b73c65418a82399cfd6feccb5297241` |
| `results/data_audit.json` | `5789b303efee5c19fcf277fd2bbaa7603041b87e1c3dd32384dac6f90d6675dd` |
| `results/bert_audit_full.json` | `0556a5d8effd6cb4cfd1f338dbd671ffaf17d6f208740a6e5694c7032127683b` |
| `results/reader_check.json` | `5c107be9d10944c71676604d5c8ac79fe2727e8f1ec86df50823098beac876b0` |
| `results/overlap_manifest_verify.json` | `9b8e807b8f8c0e51b25b235828d1ff2026595de8d77629488955572249a1c6c1` |
| `results/label_consistency_audit.json` | `eb69426f777d1f6436a06cc37bb3406c9f44cad4d1180f38373c020e7e5d98ab` |
| `results/mask_audit.json` | `5c9bd6b0fd9b2cd171e669a3bc117fd3c98bce33d660ed0eebe1fe80e905264f` |
| `results/video_audit_detailed.json` | `1d5366d6389084862f4238e9e2e3d3cff48c85da6d6be0042101d866499d8cdf` |

> 上表取自 `results/audit_run_report.json`（`status=PASS`、`total_elapsed_seconds=91.95`、11 步全 PASS）所记录的**同一次权威运行**；随后又执行 `python -m src.tools.verify_raw_integrity`（退出码 0，Gate 2）。含时间戳的 JSON 每次运行哈希不同，比对时请以该 `audit_run_report.json` 为入口；确定性产物（zip、清单、交集 CSV、接口合同）哈希稳定。`STATUS.md` 中的哈希同样由该次运行动态写入。

## 6. 未决事项（不作为通过依据）

1. 附件一容器标称帧数差异根因未证实（需 atom/NAL 解析）。
2. 音频流完整/无截断未证实；2 条 >5% 偏差样本仅记录实测值。
3. 说话人独立性未知（数据无说话人字段）。
4. 74 维音频 / 35 维视觉特征逐维语义未知，未命名。
5. 词级真实时间映射未取得 → 相关位置 `unresolved`。
6. 50MB 提交包为有依据的估算，非实测打包；冻结 BERT 权重（~440MB）不进包，外部下载是否被官方规则接受**待官方规则核验**。
7. T1 待实现：Dataset/DataLoader 的 dtype 转换、`torch.isfinite` 逐批检查、训练时缺失掩码模拟。

## 7. 复现提示

- 主入口：`& "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.audit --config configs/base.json`（工作目录 `E题\工程`）。
- 复验失败语义：`results/t0_before_fix/run_final_verification.ps1`（前置门禁 → 副本失败注入 → 权威审计 → 后置门禁 → 写回执）。
- 注意：`src.tools.verify_raw_integrity` 的产物固定写入 `results/codex_review/`（由脚本自身规定），该目录属独立复验证据区，请勿删改既有 JSON。
- 环境备注：本会话最初处于受限文件沙箱时，venv Python 写入 `results/codex_review/` 曾报 `PermissionError`；切换为完全访问后同一命令退出码 0。该现象与数据/代码无关，记录以便复验时区分。

**交回 Codex 独立复验；在复验通过前不启动 T1。**
