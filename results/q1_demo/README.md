# 第一问三模态时间展开演示

本演示只检查**问题一**的附件一视频特征及时间对齐。这里生成的特征和图不作为问题二、三的训练、验证、调参或模型输入；问题二、三使用题面规定的附件二等数据。

运行环境：从 `E题/工程` 目录，用 `E题/训练demo/.venv/Scripts/python.exe` 执行：

```powershell
& '..\训练demo\.venv\Scripts\python.exe' -m src.q1_timeline_demo --seed 23
```

更换 `--seed` 可随机抽另一条样本；加 `--sample-id=<样本ID>` 可指定视频；加 `--include-review` 可将待人工复核样本加入抽样。默认仅从有人脸且质量状态为 `PASS` 的样本中抽取。输出 PNG、SVG 和 `last_demo.json` 均在本目录。

图的横轴为视频开始后的秒数。三行依次显示 BERT 文本 768 维、音频 70 维、面部视觉 61 维。每列对应给定文本中的词及其强制对齐时间；同词多个 BERT 子词先取均值。三行共用词时间区间。颜色为**该视频内部、逐维标准化**后的相对变化，不能据此比较不同视频或三种模态的绝对值。灰色表示该时间没有有效的词级特征，包括语音间隙和被掩码的低可信对齐。图只是已提取特征的展示，不是对齐准确性的证明。

全量输出在 `data/processed/q1/`；`results/q1_feature_inventory.csv` 是逐视频状态，`results/q1_review_queue.csv` 列出需检查的词和人脸问题，`results/q1_validation.json` 给出结构复核结果。当前 100 条均已生成；结构检查通过，37 条 `PASS`、63 条 `REVIEW`。`REVIEW` 仍有特征文件，但对应词或视觉片段需要结合原视频人工核查。
