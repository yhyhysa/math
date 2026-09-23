# 2026年研赛E题 工程代码与复现说明

本项目为 2026 年第二十三届中国研究生数学建模竞赛 E 题（复杂场景下多模态情感预测的数学建模与算法设计）的工程实现代码库。

> GitHub 仓库包含代码、配置、说明、小型结果报告和问题一的 100 份派生特征（`data/processed/q1/`）。`data/raw/` 原始附件、其他处理数据、`models/` 预训练权重、`vendor/` 本机安装包、`checkpoints/` 和运行日志不上传。下列 `D:/...` 路径是原开发机示例，克隆后需按本机位置调整；没有原始附件和模型文件时，特征提取命令不能直接运行。问题一提取的特征不用于问题二、三训练。

**队友从这里开始**：见 [TEAMMATE_GUIDE.md](TEAMMATE_GUIDE.md)，其中列出 Git 必备文件、克隆后可直接运行的 Q1 结果检查，以及缺少原始附件和模型时的复现边界。所有可运行程序与可视化 Demo 的详细命令、参数、输出和读图说明统一见 [运行使用说明.md](运行使用说明.md)。下文主要是早期 T0 运行记录；`STATUS.md` 也保留为 T0 历史回执。问题一当前全量状态见 `results/q1_validation.json`。

## 一、运行环境

- **操作系统**：Windows 11
- **GPU 硬件**：NVIDIA GeForce RTX 5070 Ti (16 GB VRAM)
- **CUDA 版本**：12.8
- **Python 环境路径**：`D:/第二十三届中国研究生数学建模竞赛 - 中文题目/中文题目/E题/训练demo/.venv/Scripts/python.exe`
- **核心依赖版本**：
  - PyTorch: `2.7.1+cu128`
  - Transformers: `5.17.0`
  - PyAV: `18.1.0`
  - OpenCV: `4.11.0`
  - SciPy: `1.18.1`
  - Pandas: `3.0.1`
  - NumPy: `2.3.5`

## 二、工作目录规范

所有命令均须在 `E题/工程/` 目录下执行，使用虚拟环境的绝对 Python 路径：

```powershell
# 切换至工程根目录
cd "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\工程"

# 统一 Python 解释器调用示例
& "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m <module_name>
```

## 三、T0 阶段复现与审计命令

> 所有命令的**工作目录必须是 `E题/工程/`**，Python 一律使用上方的虚拟环境绝对路径。
> `src.audit` 主入口在**任何子检查失败时非零退出**，且**不写 PASS、不覆盖已有 STATUS**；失败时只写 `results/audit_run_report.json` 并把 `STATUS.md` 标为 BLOCKED。

### 1. 运行全量 T0 审计主入口（含原始完整性前后门禁）

```powershell
& "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.audit --config configs/base.json
```

主入口内部依次执行：完整性门禁(Gate 1) → 统一读取器 → 245 行清单 → 跨附件交集 → 交集/清单独立复核 → 标签一致性 → 全量 BERT → 掩码/dtype 统计 → 视频审计 → 完整性门禁(Gate 2)。

> `--skip-integrity` 为**测试专用**开关（跳过门禁），仅用于在隔离副本中测量“子检查失败”的退出语义，正常复现不要使用。

### 2. 独立运行各子项检查工具

- **原始完整性核验（只读，写入 `results/codex_review/`）**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.verify_raw_integrity
  ```

- **数据读取器全文件断言测试**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.check_data_reader
  ```
  输出：`results/reader_check.json`

- **跨附件精确交集重算**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.cross_overlap
  ```
  输出：`results/cross_attachment_overlap.csv`

- **交集 + 245 行清单独立复核（从原始输入独立重算并与交付产物比对）**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.verify_cross_overlap
  ```
  输出：`results/overlap_manifest_verify.json`

- **附件一 / 附件二 重叠样本标签与转写一致性**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.audit_label_consistency
  ```
  输出：`results/label_consistency_audit.json`

- **全量 245 行输入资产清单生成**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.build_manifest
  ```
  输出：`results/input_manifest.csv`

- **全量 4870 样本 BERT 重构误差分布评估（绑定固定 revision 快照）**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.audit_bert_full
  ```
  输出：`results/bert_audit_full.json`

  仅校验快照绑定（不推理、不写产物）：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.audit_bert_full --self-check
  ```

- **padding / 全零特征 / dtype 实测统计**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.mask_audit
  ```
  输出：`results/mask_audit.json`

- **详细原视频双解码器与音频时长偏差审计**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.audit_videos_detailed
  ```
  输出：`results/video_audit_detailed.json`

- **安全解压（已有文件只读，绝不覆盖）**：
  ```powershell
  & "D:\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\E题\训练demo\.venv\Scripts\python.exe" -m src.tools.safe_extract --zip data/raw/E题数据.zip --out data/raw/extracted
  ```
  已有文件按内容哈希（CRC32 + SHA-256）比对：一致则跳过，不一致则**报错并退出码 2，不覆盖**。

## 四、冻结文本编码器的复现边界

- 模型：`bert-base-uncased`，**固定 revision `86b5e0934494bd15c9632b12f734a8a67f723594`**。
- 加载方式：从本地快照目录 `snapshots/86b5e0934494bd15c9632b12f734a8a67f723594` 加载，`local_files_only=True` + `HF_HUB_OFFLINE=1`，**不解析可变的 `refs/main`**；加载前后逐文件核对 SHA-256，缺失或哈希不符即非零退出。
- 记录哈希：`vocab.txt`=07eced37…、`config.json`=7160e155…、`model.safetensors`=68d45e23…、`tokenizer.json`=ce64fce7…（完整值见 `results/bert_audit_full.json` 的 `model_metadata.file_hashes`）。
- **不宣称完全无损**：报告只覆盖被实际评估的样本（附件二 4850 + 附件四 20）；附件三无 `raw_text`，只核验形状/token 范围/特殊 token/冻结编码器可运行。
- 权重（~440MB）不进入 50MB 提交包；外部下载是否被官方规则接受**待官方规则核验**。

## 五、工程目录结构

```
工程/
├── README.md                      # 本说明文档
├── STATUS.md                      # 阶段状态；由 src.audit 按本次运行动态生成
├── configs/
│   └── base.json                  # 基础配置与路径映射
├── data/
│   ├── raw/                       # 原始数据压缩包与解压目录 (只读)
│   └── processed/                 # 预处理与缓存特征目录
├── results/                       # 审计报告、清单与接口契约
│   ├── data_audit.json            # T0 全量主审计报告（含各步骤退出码与耗时）
│   ├── audit_run_report.json      # 每次运行的步骤表/哈希；失败时记录失败步骤
│   ├── input_manifest.csv         # 245 项输入资产 SHA-256 清单
│   ├── interface_contract.json    # 多模态统一接口与模型契约
│   ├── reader_check.json          # 51 个真实 pkl 读取器断言测试
│   ├── cross_attachment_overlap.csv # 附件一与附件二 18 个重叠样本
│   ├── overlap_manifest_verify.json # 交集与清单的独立复核结果
│   ├── label_consistency_audit.json # 18 条重叠样本标签/转写一致性
│   ├── bert_audit_full.json       # 4870 个样本 BERT 重构误差统计（含快照绑定）
│   ├── mask_audit.json            # padding/全零/dtype 实测统计
│   ├── video_audit_detailed.json  # 100+20 视频流详细审计表
│   ├── codex_review/              # 独立复验证据（原始完整性 JSON 等）
│   └── t0_before_fix/             # 历次返工前的产物与源码快照
└── src/
    ├── audit.py                   # T0 审计主调度入口（失败即非零退出）
    ├── data.py                    # 统一多模态数据加载与解包器
    ├── config.py                  # 配置解析与绝对路径适配
    ├── check_data_reader.py       # 读取器自动化测试套件
    └── tools/                     # 各专项审计与清单生成工具
        ├── verify_raw_integrity.py    # 原始 zip/解压/清单完整性核验
        ├── verify_cross_overlap.py    # 交集与清单独立复核
        ├── audit_label_consistency.py # 附件一/二标签一致性
        ├── cross_overlap.py           # 交集重算并写 CSV
        ├── build_manifest.py          # 245 行清单生成
        ├── audit_bert_full.py         # BERT 全量重构误差 + 快照绑定
        ├── mask_audit.py              # padding/全零/dtype 统计
        ├── audit_videos_detailed.py   # 视频/音频流详细审计
        └── safe_extract.py            # 只读安全解压（绝不覆盖）
```

## 六、结论边界（T0 不越界声明）

- 附件一容器标称帧数差异的**根因待确认**（未解析 MP4 atom / H.264 NAL）。
- 音频流**不宣称完整**；仅实测可解码时长与容器时长偏差，2 条 >5% 样本已单独记录。
- **说话人独立性未知**（数据无说话人字段）。
- **74 / 35 维特征逐维语义未知**，不猜测命名。
- **词级真实时间映射未取得** → 相关内容标 `unresolved`。
- 50MB 提交包为**有依据的估算**，非实测打包；冻结 BERT 外部下载是否被官方规则接受**待核验**。
- Dataset/DataLoader 的 dtype 转换、`torch.isfinite` 逐批检查与训练时缺失掩码属 **T1 待实现**，T0 未执行、不宣称已执行。
