# 给队友看的工程交接说明

## 哪些文件必须留在 Git 中

| 路径 | 用途 |
|---|---|
| `src/`、`configs/` | T0 数据审计、问题一特征提取、结果核验和绘图代码，以及对应配置。 |
| `data/processed/q1/` | 用户确认上传的 100 条问题一派生特征：每条 1 个 JSON 元数据和 1 个 NPZ 特征文件，共 200 个文件。 |
| `results/q1_feature_inventory.csv`、`results/q1_validation.json` | 逐样本状态与全量结构检查。 |
| `results/q1_quality_proxy_metrics.csv`、`results/q1_quality_proxy_report.json` | 自动词对齐覆盖率、人脸检出率及其明确口径。 |
| `results/q1_review_queue.csv`、`results/q1_manual_review_sheet.csv`、`results/q1_manual_review_instructions.md` | 待人工复核位置及填写说明。人工判定尚未完成。 |
| `results/q1_demo/`、`results/q1_review_figures/`、`results/q1_quality_proxy_dashboard.*`、`results/q1_face_threshold_probe/` | 三模态热图、四层原视频审查图、覆盖率图及人脸检测参数试验。 |
| `Q1三模态特征与时间对齐方案.md`、`README.md`、`运行使用说明.md`、本文件 | 方法、统一运行方法、运行边界和交接说明。 |
| `AGENTS.md`、`.gitignore`、`.gitattributes`、`requirements-review.txt` | 原始数据保护、提交范围和已上传结果的轻量读取环境。 |

`results/` 中其他小型 T0 审计报告也保留作历史证据。`STATUS.md` 是早期 T0 执行回执，**不是问题一的最新总状态**；问题一以 `results/q1_validation.json` 和本说明为准。

## 哪些文件不上传

`data/raw/` 及其他原始附件、`models/` 预训练权重、`vendor/` 本机依赖、`checkpoints/`、`submission/`、虚拟环境、缓存和大日志均由 `.gitignore` 排除。尤其不要为了让代码“能跑”而把原始视频、附件二至四或模型权重直接提交到 Git。当前 Q1 派生特征约 17.4 MB；整库工作树中的这些外部数据和模型不在仓库里。

## 克隆后立即可做的事

在仓库根目录执行，示例使用 Windows PowerShell 和 Python 3.12：

```powershell
git clone https://github.com/yhyhysa/math.git
Set-Location math
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements-review.txt
& .\.venv\Scripts\python.exe -m src.q1_validate
& .\.venv\Scripts\python.exe -m src.q1_quality_dashboard
& .\.venv\Scripts\python.exe -m src.q1_timeline_demo --seed 23
```

`src.q1_validate` 应检查 100 个样本、200 个 Q1 文件的形状、掩码、窗口覆盖和特征 SHA；当前结果为 37 条自动 `PASS`、63 条 `REVIEW`、0 条程序失败。绘图命令会重生成自动覆盖率图和随机时间线图。`PASS` 只是自动规则通过；词边界和目标人物仍须回看原视频。

逐个程序的完整命令、参数、输入输出和读图方法请看 [运行使用说明.md](运行使用说明.md)。其中四层审查图需要原始 MP4，人脸试验还需要本地 MediaPipe 模型；仅克隆仓库不能直接重画原始视频帧。

## 重新从原视频提取时的额外条件

克隆仓库**不能单独重跑** `src.q1_extract` 或完整 T0 原始附件审计。需另外准备题目授权的原始附件，放在代码所期望的 `data/raw/extracted/E题数据/` 结构中；准备 `models/bert-base-uncased/86b5e0934494bd15c9632b12f734a8a67f723594/`、`models/wav2vec2-base-960h/` 和 `models/face_landmarker.task`；还需 PyTorch、Transformers、PyAV、openSMILE、MediaPipe、librosa、openpyxl、SciPy 等提取环境。当前 `q1_extract.py` 使用本机 `vendor/opensmile/core/config` 相对路径，需要在另一台机器配置对应的 openSMILE 配置目录。原仓库的 `configs/base.json` 与旧 README 中有原开发机绝对路径，迁移时也要改成本机路径。不要把现有 JSON 中记录的原开发机路径当成队友机器上的有效文件路径。

问题一的 70 维音频、61 维面部视觉特征只用于问题一；问题二、三按赛题使用附件二等规定的数据，不读取这里的 Q1 特征训练、验证或调参。
