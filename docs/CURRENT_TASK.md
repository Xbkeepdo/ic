# 当前任务

生成与修正评分均已完成（2026-11-17）：Qwen3 SADI＋当前层→下一层，新200图，max_new_tokens256。400条描述未重跑。独立审计发现旧CHAIR整段Treebank分词漏计多句文本中chairs.等物体词，已在独立目录进行Punkt分句→逐句Treebank的CPU重评分，GT与预测同时修正、新建GT缓存，原文件哈希未变。当前最终结果必须使用修正口径。

修正主结果（CHAIRs／CHAIRi／召回／F1）：SADI23.00／4.878／71.73／75.93%；next21.00／4.869／71.49／76.09%。幻觉图46→42，修复8/新增4；幻觉提及86→80，总提及1763→1643。四主指标95%配对区间均跨0：CS[-5.50,+1.50]pp；CI[-1.197,+1.129]pp；召回[-1.555,+1.033]pp；F1[-1.170,+1.481]pp。CHAIRs温和正向点估计，CHAIRi基本无改善，不能宣称稳定显著收益。

最大减少图187169的11→4（table重复10→3，sailboat均1）超过全量净减少6次。去掉此图199张：CHAIRs22.61→20.60%，CHAIRi4.29→4.64%反转，幻觉75→76。该图next原漏计chair1已恢复，SADI chair4不变；新GT缓存重建后200图GT物体集合均无变化。共同未触顶99图修正CS18.18→16.16%、CI5.31→5.05%、F175.37→75.87%；112图B16修正CI3.63→3.77%；均补充、区间跨0。

最终报告：docs/archive/QWEN3_SADI_NEXT200_T256_SENTENCE_EVAL_20261117.md。原冻结结果存档：docs/archive/QWEN3_SADI_NEXT200_T256_20261117.md（旧CS22→20，CI4.78→4.30等不能继续作为修正后结论）。本次只修正当前200；旧100/500尚未重评，不能混用评估口径。

服务器生成根目录：/root/rivermind-data/ic-ac/outputs/generation_time_evidence_drift_qwen3_next200_t256_b16/。修正输出为该目录下sentence_tokenized_evaluation_v1/，含coco200/metrics.first200.sadi-sadi_sar_ems_next_19_20.json、逐图分数、新GTcache、tokenizer_fix_audit.json、microtests.json、provenance.json。独立corrected_audit.json全部复核PASS，400raw SHA、GT集合、主指标及bootstrap/补充/分母排除均一致。本地相同结构在C:/Users/xbo/AppData/Local/Temp/qwen3_next200_t256_checked/。原评分/400 raw/旧cache/全部冻结源码前后SHA相同。修正脚本scripts/evaluate_qwen3_next200_sentence_tokenized.py；不改旧CHAIR源码，保持原POS/lemma/同义词/指标公式及绝对字符偏移，单句/Unicode/中间句末微测通过。

生成协议：SADI零起5–18；next Q18/P18→A19、Q19/P19→A20；强度5/tau0.03/noun+positive-gain固定。greedy、BF16、Describe this image.，两臂显式256。2卡各100，33个匹配batch，max16，实际112图B16；无OOM/Traceback。两组各90图达256。198图触发、2352生成步/2832逐图逐层事件、140序列改变，无同层重算。

seed43排除旧5350图（包括前100），35154候选抽200，与9来源零重合。ID SHA3e59a3aa06cffca8e671fe3a4467ccb8bc96f353c98d293139abf963c168ab1d，23源码hash冻结，13推理源码/config复用旧GPU验证未重测。快照scripts/experiment_snapshots/qwen3_next200_t256_b16/，原controller/generate_group不改。local/server四旧文件rawhash差异仅换行，规范文本相同，server字节副本已存server_sources。

生成北京时间07:05:56→07:17:06约11分10秒，原评分07:17:19完成约11分22秒。后续额外仅CPU评分，无GPU重生成。两臂累计GPU进程生成611.96/664.57秒（next+8.6%，受长度影响、不是墙钟）。本轮源码、配置、实验快照和报告已同步到 GitHub 的 `mitigate` 分支。复现入口见 [缓解实验说明](MITIGATION.md)。旧历史见docs/archive/CURRENT_TASK_before_next200_t256_20261117.md。

CPU探针首次尝试PunktTokenizer新API不适配服务器旧NLTK，改nltk.data.load('tokenizers/punkt/english.pickle')成功；未修改依赖。所有原始与修正评估产物分别保留。
