# CHAIR评估器选项

新增入口要求显式选择版本，不自动切换：

- `--evaluator legacy`：保留原整段Treebank分词，复现历史评估口径。
- `--evaluator sentence`：先Punkt分句再Treebank分词，修正多句描述中部分句末物体词漏计。预测与GT captions同时使用该处理。

原评估脚本和历史分数保持原样；新增入口没有改变它们的默认行为。旧版已知会漏计部分带句末标点的物体词，比较和汇报时须标明版本。

服务器上复现本次Qwen3 200图：

```bash
cd /root/rivermind-data/ic-ac
/root/rivermind-data/envs/ic-feature-extract/bin/python scripts/evaluate_caption_chair.py \
  --evaluator legacy \
  --generation-root outputs/generation_time_evidence_drift_qwen3_next200_t256_b16 \
  --cohort coco200 --variants sadi sadi_sar_ems_next_19_20
```

把`legacy`改为`sentence`即可使用修正版。默认分别写入生成根目录下的`evaluations/legacy_v1/`和`evaluations/sentence_v1/`，原始描述与旧评分不覆盖。可用`--evaluation-root`指定其他独立目录。

每份主指标JSON写入`evaluator`和`evaluation_inputs`，GT缓存写入同一评估器标识与源码/资源哈希。不同版本的缓存不混用；同目录版本或源码不匹配时拒绝复用。原生成文件只读复用，整个cohort输出目录不会链接回原目录。`--limit`和`--bootstrap-draws`沿用共享评估器含义，后者默认5000。

两种口径应分别汇报。同一个比较中的基底和增强方法必须采用相同版本。`sentence`是本仓库的分句修正版本，并非声称与原论文官方CHAIR实现逐项等价。

2026-11-17服务器CPU回归通过：用同一已保存200图分别运行两选项，逐图评分、四项指标及全部5,000次配对区间分别与原旧版和修正版完全一致。省略`--evaluator`会报参数错误；跨版本复用同一评估目录会被拒绝；原描述、原指标、旧GT缓存与冻结源码哈希均未变化。验证记录位于生成根目录`evaluations/selector_verification.json`。未运行GPU生成。
