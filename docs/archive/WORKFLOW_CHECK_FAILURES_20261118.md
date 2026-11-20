# 工作流检查记录

- update_ffn_alignment_queue_state.py首次生成的状态为8305字节，触发8KB断言，原CURRENT_TASK未被覆盖；已归档原状态并压缩历史摘要，修正后通过。模型运行及实验协议不受影响。
- PowerShell 下向 rg 直接传入 `docs/archive/QWEN3*EXPAND*` 等路径通配符，未展开导致 os error 123；改为 `rg -g '*.md' ... docs/archive`，读取成功，无实验影响。
- 第四批500 CPU准备检查：从 `/tmp` 执行检查脚本未加仓库到 `sys.path`，导入失败；已补充路径。随后继承的源码清单混用绝对/相对路径，旧驱动对 `run_qwen3_signed_feedback.py` 的既有相对路径兼容条件未命中；现按既有驱动将仓库内源码统一为相对路径，同时仍逐项验证当前100图冻结SHA。两次均在启动GPU前失败，未放宽算法源码校验。
- 第四批500首次组合审核在“同文字前缀必须同原目标”处失败。定位226128：step60已保留干预而token不变，step71缓存状态已不同，故旧条件不成立。新增retained-prefix规则及v2审核，只在首次更早保留前保证原生状态，并同时加强前缀相等要求；其他物理／类别／概率／回滚条件全保留。新增CPU反例验证及六批SVAR轨迹重验通过。生成、阈值、指标和旧冻结源码未改；旧失败状态保留，恢复状态另存。详第四批报告及audit_revision_protocol.json。
- prefix恢复状态摘要首次为8269字节，8KB断言在写入前拦截；压缩已归档历史说明后写入成功，实验数据无影响。
- 读取旧临时helper `svar_midpoint500_qdest_sensitivity.py` 时路径不存在；该旧定位未用于新实验，改为直接读取仓库中的通用提及/分母分析函数。模型运行与数据均未受影响。
- 多层FFN选择的首次CPU验证在原19层梯度逐值相等处失败，诊断最大差5.96e−8：将父式两个view改为共享view后，梯度求和位置发生变化。恢复父式两个view并对原leaf求导后严格相等通过，四层两类梯度独立有限差分误差0.000223398；没有放宽原断言。发生在新GPU实验启动前，未影响任何既有生成或冻结源码。
- 层选择终局状态摘要首次8545字节，8KB断言在覆盖CURRENT_TASK前拦截；已保留旧状态归档并改为压缩摘要。全部实验与归档数据未受影响。

- 门控覆盖审计读取时再次直接给rg传features/qwen3_*.py，Windows返回os error123；改为目录加-g过滤后成功，未运行或修改模型。

- semantic-final GPU预检v1的正常两图32token未触发，拒绝路径的liveness断言失败。保留原状态和源码，新增仅供预检的强制正gain控制，正式门控未改。v2在`inference_mode`内部加载模型，参数无版本计数而静态cache版本检查失败；改为和正式worker相同在该上下文外加载模型。v3通过关闭、零重算、强制拒绝、正常、强制两语义版本与空间置换测试。failed_v1/v2日志/源码均保留；正式100在通过后才启动。

- semantic-final结果首次SFTP下载时，Windows本地目标路径加五方法组合指标文件名超过传统路径长度，FileNotFoundError；远端生成、分析和三项审核已经成功，未重跑模型。最终归档将长产物路径映射为短哈希文件名，保留原路径及逐SHA映射。一次工具参数解析失败未执行命令，后续使用明确的原始多行字符串执行成功。

- semantic-final首次收尾在review_image_provenance.json的SHA断言失败：本地Python写入CRLF、SFTP文本写入LF，JSON内容一致。保留远端原LF文件，逐字节确认除换行外完全相同后改为二进制上传，使人审引用的原SHA与两端文件一致；未放宽校验，第二次216项及全部输入核查通过。失败发生在归档创建前，模型、协议、评估均未改变。

- risk_F只读trace的100图生成和特征读取全部完成，v1分析因错误要求“查询首token别名类别=完整CHAIR提及类别”失败：skateboarder→skateboard/person、Carrots→car/carrot。原协议、源码、原failed状态保留；新增v2分别核对token别名与完整提及，191主位置不删，另列189位置敏感性。全部209在线历史条件及门控概率重验通过，CPU最大差1.19e-7。没有修改阈值/模型/标签或重新运行GPU。

- risk_F收尾状态首次8414字节，8KB断言在覆盖CURRENT_TASK前拦截；旧状态已归档，精简历史和下一步说明后写入成功。模型、数据、归档及协议无影响。

- 新语义donor纯张量原型的首次CPU验证在“支持全为1时预算必须精确0”失败：1−归一化权重和有1.1921e-7舍入残差，最大delta4.394e-9。旧源码/失败元数据保存failed_v1；改为直接加权1−支持度并处理全零写入，保持严格零断言后通过。历史控制逐值相同、置换逐头mass差9.31e-10均过。尚未接入解码器，没有模型生成或旧冻结产物被更改。


## Semantic donor GPU preflight v1

独立预检进程684176在首个实际提案因KeyError(19)退出，未启动100图。移除不必要的readout_contrast后，原先在该函数内隐式创建的FP32 o_proj缓存没有初始化。已在新控制器中显式创建该投影；失败日志、请求SHA及3份原源码保留在outputs/qwen3_semantic_donor_preflight/failed_v1_sources。修正版使用独立outputs/qwen3_semantic_donor_preflight_v2，不覆盖失败记录或旧冻结依赖。


## Semantic donor development100 OOM and memory recovery

首轮实际100主进程684532/worker684535/684536终局，run_status保留failed。GPU0后续组在repeat_kv申请72MiB时OOM；已分配29.45GiB，保留未用1.68GiB。原生读取器与owner相互引用形成Python环，结束组后GPU张量没有及时释放，内存随组累积。新增独立resume_qwen3_semantic_donor_pilot.py，每次生成后gc.collect/empty_cache，并记录实际清理前后内存。原151冻结依赖完全不改；旧失败目录保持，memory_recovery新目录保留完整输出前缀（history54图、semantic41图），原有序批次及B、模型、公式、门控、接受规则不变，跳过完成组，不重跑。每卡首次恢复组额外检查原32-token基线；结束必须验证旧输出字节前缀一致。


## ACG后层适配预检与检索

首次GPU预检685958在zero_replay分支退出：父类零预算形状是[B,H,K]而非2D，新类将其误当作[B,H,D]上下文修正，未经过_budget因此无acg_context。v1请求、3源码、日志和CPU证明已保留；修正版按diagnostic_zero_replay显式转给原注意力零重算，方法公式未改，CPU_v2再次passed，使用独立preflight_v2。未启动实际100。Windows检索误用scripts/run*路径报os error123，改为目录配合-g正常完成；CVF CausalLens PDF网页403，仅ACG原文用于当前实现。


## Joint ACG snapshot path resolution

首次TEMP/block_acg_finalize.py在核缓存证书的相对路径时，从SSH默认home而非仓库解析，FileNotFoundError；在写final_provenance和建tar前退出。仅在归档进程显式os.chdir(repo)后重试，全部相对路径SHA通过。没有修改方法、冻结依赖或重跑GPU。


## Joint ACG head-permutation cached records

首轮driver689150在96张记录后因旧source_groups只接受native decoder而拒绝显式缓存记录；保存原failed目录和全部96行。首次恢复689598在prepare校验缓存证书时因证书relative键与source绝对路径不一致KeyError，未运行GPU；源码与failure.json保存。独立v2数值审计按仓库解析路径后仍逐SHA核同一源，CPU验证原96行/9缓存全过；新recovery_v2只补原GPU0最后4图，原96行逐字节保留。原方法及7冻结源未改。

首轮事后mechanism审计误把raw full事件与父类scoped事件直接逐字典比较，后者多generated_token字段；在2179确认唯一差为generated_token=1965。原失败status/log保存，新v2独立审计将其与full emitted_token精确对应，其余字段保持全量相等，没有放宽数值/接受条件，未重跑GPU。

一次主报告无实质内容的补丁使用了不完整上下文，apply_patch未找到段落而拒绝；文件未变，改用已核对内容追加封包后校验记录。

头对照收尾CURRENT_TASK首次8365字节，被8KB断言拦截且未覆盖旧状态；旧状态已归档，压缩专门报告已保存的审计细节后写入。一次修复编排因PowerShell换行符与JS模板边界冲突在执行前报语法错误，改用Python处理后成功。实验与冻结源无变化。


独立500分析收尾状态首次8840字节，8KB断言在覆盖前拦截。旧状态已归档；将已在报告保存的旧100/头对照细节压缩，保留活跃进程、冻结源和下一步审计后成功写入。私有终局helper已加第4步分析但未执行；模型/参数/图像未变。


更新独立500进度时，一条Python import误直接传给PowerShell，命令在执行前失败。已改用私有Python payload更新记录，未修改实验、冻结代码或运行进程。


2026-11-19 independent500 CPU case formatter: analyze_qwen3_block_acg_fresh.py raised KeyError(predicted_log_probability_change) on fixed no-gradient proposals. Three numeric/mechanism audits already passed. Preserved frozen source, failed log/status and raw captions; separate v2 formatter and recovery protocol permit null only for the fixed branch. Complete448-event check and CPU-only reanalysis passed; no model rerun.
2026-11-19 collective pilot prelaunch: remote py_compile caught an extra closing parenthesis in the new analysis-protocol writer. No output root or GPU process was created. Removed the parenthesis, recompiled locally, then relaunched once.

2026-11-19 collective snapshot local verification: generic renaming produced nonexistent verify_qwen3_collective_visual.py in the local active-source list. Remote362-item archive, transfer, tar SHA and every entry had already passed. Corrected the local list to verify_qwen3_collective_visual_gate.py, verified seven real active sources, and wrote snapshot_archive.json. No archive, model or analysis rerun.

2026-11-19 bounded pilot CURRENT_TASK update exceeded the8000-byte documentation guard. A follow-up replacement initially used the old paragraph delimiter; re-read the current text, condensed completed history, checked length before writing, and resynced. No code/protocol/model/audit changed.

2026-11-19 bounded pilot mechanism audit v1 failed on bit-exact fixed coefficient mean. All629 layer records have drift1/min=max0.5;34 means differ by2.9802322387695312e-08 (FP32 reduction rounding). Original generation/audit source and failure logs retained. Separate v2 changes only this mean equality to one FP32 ULP at0.5 (2^-24); extrema and every other criterion stay strict. CPU recovery will reuse all raw captions and both evaluations.

2026-11-19 bounded500 read-only performance inspection: rg was given a nonexistent models/generation_utils.py and a Windows wildcard path models/*.py. Corrected to rg -g "*.py" over models; found configure_image_processor_limits in base_wrapper.py. No experiment source, settings or process changed.

2026-11-19 bounded500 closure CURRENT_TASK draft8288 bytes exceeded the8000-byte guard before state overwrite. The archive footer had already been written locally. Condensed completed/preflight history references and verified that the report body still starts with the sealed report_before_archive.md; no scientific source, raw output, evaluation or444-item bundle changed.

2026-11-19 causal-tilt preflight preparation: Paramiko SFTP mode x created an empty CPU-source snapshot but did not open it for writing (OSError). No GPU request/root/process was created; the actual remote feature remained at the CPU-checked version. Preserve the empty file as .failed_empty and use exclusive write mode wx before the single GPU launch.

2026-11-19 causal GPU preflight attempt1: both actual workers terminated with missing wordnet because NLTK_DATA was omitted. ObjectClassHistory raised before verify() and any model.generate call. Preserve request/launch/B2.log/B16.log and attempt1_failed.json; environment_recovery_request.json adds the existing cache path only. All325 frozen inputs unchanged, same tests/parameters, attempt2 logs and PIDs separate; no resource download.

2026-11-19 causal pilot preparation: PowerShell rg path features/qwen3* was not expanded (os error 123). Reissued rg against features with -g 'qwen3*.py'; read-only failure, no experiment affected.

2026-11-19 causal100 reporting: inspecting archived old100 metrics with an original-filename glob raised StopIteration because the tar archive mapped long paths. Read independent_summary.json instead; no evaluation or GPU operation repeated.

2026-11-19 causal500 preparation: one batched rg included a guessed non-existing features/qwen3_caption_patch_drift.py and returned os error 2. Re-read the actual batched generation/controller files. Read-only failure; no model or experiment repeated.

2026-11-19 gate review: rg with literal PowerShell wildcard paths returned os error 123; corrected to directory plus -g filter. A CURRENT_TASK draft exceeded 8 KB and the guard rejected it before writing; shortened after archiving. No experiment affected or restarted. New mass-matching CPU proof passed first run.

- 2026-11-19 mass-matched pilot preparation: parent workspace Get-Content AGENTS.md returned not found; actual repository is ic-ac and its AGENTS.md was read before edits. No experiment or frozen source changed.

- 2026-11-19 mass-matched closeout preparation: rg --files scripts/experiment_snapshots/qwen3_causal_visual_tilt_pilot_20261119 used an incorrect date suffix. Local snapshot is qwen3_causal_visual_tilt_pilot; discovery under the snapshot directory located it. Read-only lookup failure, no experiment effect.

- 2026-11-19 mass-matched denominator diagnostic v1: a field named class multisets compared raw (surface word, canonical class) pairs, reporting false due to wording changes. Preserved v1; v2 separately verifies raw-pair inequality and canonical-class equality. No captions, scores, candidate parameters or numerical selection rules changed.

- 2026-11-19 semantic gate investigation: rg with PowerShell positional wildcard paths QWEN3_FULL_RISK_F* etc returned invalid filename syntax. Reissued on docs/archive with -g filters and read relevant files successfully. Read-only lookup failure; CPU diagnostics and new100 generation unaffected.

- 2026-11-19 risk/E_l100 closeout: first CURRENT_TASK draft exceeded the8000-byte guard and was not written; the subsequent sync sent the prior running-state summary. Replaced it with a shorter verified closed-state summary and synced again. Experiments, frozen sources and scores unchanged.

- 2026-11-19 head-routing literature review: a guessed head-gradient report filename was absent; rg located QWEN3_SAR_EMS_HEAD_GRADIENT_20261118.md. CVF CausalLens full-paper access returned403, so only the official repository implementation was used and no unavailable paper formula was asserted. A full repository-tree print was unnecessarily large; subsequent inspection filtered stored paths.

- 2026-11-19 head-routing case inspection: printing the complete per-method scores also expanded embedded event statistics, producing oversized output. Switched to an explicit scalar-field whitelist. Raw captions, scores, protocol and all four successful audits were unchanged.

- 2026-11-19 route exploration read-only inspection: guessed semantic_final_target.py and probe_history_route.py were absent; the already-listed semantic_final_feedback.py and sar_ems_history_route.py were read instead. The first rg exclusion missed nested experiment_snapshots; corrected to !**/experiment_snapshots/**. No experiment or frozen source changed. ACD's official PDF returned an unsupported content type and OpenReview a verification page; only its official abstract was used, not an inferred full formula.

- 2026-11-19 route progress save: a worker exited normally between reading the generating status and reading /proc/PID/stat, so the live-identity save raised FileNotFoundError. The local CPU-coverage report footer had already been appended, while CURRENT_TASK was not changed. Re-polled the same pilot and confirmed complete, all four audits passed, all three processes missing. No restart or experiment change; a separate closeout sync preserves the report prefix and records terminal state.

- 2026-11-19 route terminal-state draft was8002bytes; the guard rejected it before overwrite. Condensed archived-result references and synced a7684byte state. The first result-pack SSH observation timed out during gzip of the895MB case file; later /proc inspection found no archiver and a stable325,945,113byte archive. Reused that completed archive, verified its SHA and every72entry after download; no generation, evaluation or compression rerun. Final archival uses a separate small supplement to avoid recompressing the large original results.

- 2026-11-19 mild-route preparation: rg received literal PowerShell paths scripts/evaluate* and scripts/eval*, returning invalid filename syntax. Reissued on scripts with -g filters. Read-only failure; all new CPU and B2/B16 checks passed without experimental retries.

- 2026-11-19 mild100核查：首次元数据读取把 metrics.per_image（路径字符串）误当嵌入字典，AttributeError；改按冻结 result_input_paths 读取对应逐图JSONL。未改原始结果或重跑实验。

- 2026-11-19 soft-dose探索：rg features/qwen3* 在PowerShell传入字面通配路径，报os error123；已改目录features配单个-g qwen3*.py。首次观察器沿用mild前缀只显示复制对照计数，已改只读glob soft*.jsonl；未改运行或重启。

- 2026-11-19 soft-dose CPU归档：输入清单含相对路径，relative_to绝对上游目录报ValueError；统一resolve，并核验已有小文件后续接归档，不重跑诊断/模型。

- 2026-11-19 soft100文档末尾核验：按LF分隔归档附记时残留Windows CR，正文hash断言失败；改按已封存正文的精确字节数切前缀，本机startswith已通过。未修改封存正文、结果包或分数。

- 2026-11-19 source-adaptation准备：读取 guessed full_i_el快照目录和models/qwen3_vl.py/model_factory.py时路径不存在；改按rg --files定位实际qwen3_vl_wrapper.py及既有输出目录。一次functions JS字符串转义未通过解析，修正后py_compile通过；均未启动或改变实验。

- 2026-11-19 source-cache v1误把legacy target_log_s当原始L，独立恒等式断言失败。已封存v1精确源、partial cache和failed.json；v2由已有object_probability_norm_hpre重建L并核验归一化OT目标及El，通过。未放松容差、无GPU重提取。

- 2026-11-19 parallel SADI预检：进程正常退出但8图31位置均未过预设数值容差（risk最大.46299/I相对65.02%/probe.19652）；已保存失败协议和结果，不放宽阈值。无SADI长度诊断也观察到视觉prefix差异，故不复用旧空间概率或整段teacher捷径。原生逐token采集8图通过hidden+KV精确控制后才单次启动两卡全源。

- 2026-11-19 native-source状态稿8457bytes超过8000字节守卫，写入前断言拦截，旧CURRENT_TASK未变；缩短已归档历史细节后重写。未改任何实验源或重启进程。

- 2026-11-20源门控结果CURRENT_TASK稿8719bytes超过8KB守卫，写入前拦截；已缩短历史/表格后保存。所有数据、权重、协议和运行未变。

- 2026-11-20 source-policy查阅：猜测QWEN3_GROUNDED_RIVAL_PILOT_20261118.md和features/caption_extractor.py路径不存在；改用rg --files定位实际QWEN3_LEXICAL_RIVAL_RECYCLE_20261118.md及utils/token_alignment.py。只读失败，未修改冻结方法或重跑实验。

- 2026-11-20 native-caption源准备：最早20261116 SADI500的batch/generation源码hash不符，建根前拒绝；未GPU，失败源码/记录封存dependency_preflight。保留守卫，改用最近已完成且当前基底匹配的causal_visual_tilt_validation500源，70目标全部首选匹配后才启动完整读取。CURRENT_TASK8195bytes稿被8KB守卫写前拦截；缩短已归档细节后保存，实验未变。

- 2026-11-20 native-caption原程序错误要求每批至少有目标，45组808目标后空组25/52退出。父子确认终止后独立recovery按hash复用45组，只算13剩余非空组、4空组零模型调用，数值段逐字不变；904/904匹配，fit/diag均rc0。CURRENT_TASK8387bytes稿写前拦截，缩短闭合历史后保存；另两猜测报告路径不存在，改实际TRANSFER报告，未重复GPU。

- 2026-11-20 region-consistency新脚本dict括号多一个，编译/准备失败、无protocol，launch被守卫拒绝，未开GPU；修正并加退出码守卫后各GPU阶段全过。另猜测risk_semantic_feedback文件不存在已改实际模块。CURRENT_TASK8617bytes稿写前拦截，旧文件未变，缩短已归档细节后保存。

- 2026-11-20 region-live查阅：rg路径参数使用PowerShell未展开的evaluate_qwen3_caption*，只读报错；改读实际evaluate_caption_chair.py，未改变冻结代码或实验。

- 2026-11-20 region100：两卡完整生成rc0、机制审计通过；旧评估摘要硬取guided_steps，而新控制器仅有proposals/accepted，sentence在指标落盘前KeyError。保存原失败脚本/日志与原caption，独立recovery只加活动统计schema适配；CHAIR分词、逐图评分、聚合和bootstrap不改、不重生成。另一次只读PowerShell wildcard路径报错，已改SFTP白名单读取。

- 2026-11-20 supported-rival：只读检索猜测llava_sar_ems_ffn.py不存在，改读实际qwen3_sar_ems_ffn_aux.py。全部GPU/CPU阶段通过；发现FFN日志优化step被生成step覆盖，仅新写alignment_step_metadata.json按冻结公式恢复.644244/.390076，原源码/结果不改、不重跑，后续新版本分开字段。

- 2026-11-20 CLVA预检v1：两卡在disabled首次图像缓存guard失败，均终止rc1，无CLVA caption。原生SADI128已匹配；源码确认get_image_features会castBF16，runner却用处理器float32输入做缓存键。独立v2仅从真实encoder hook参数取pixel键，严格相等检查不变，冻结v1源/失败记录，方法参数不改。VISTA官方PDF浏览器不支持octet-stream，未据此声称读过正文。

- 2026-11-20 CLVA100状态更新：同一apply_patch对CURRENT_TASK同时Delete/Add被工具拒绝，整个调用未执行；改用临时文本并在字节／行数守卫通过后写入，5477B／25行。未改实验代码、协议或进程。

- 2026-11-20 CLVA质量补查：一次rg把features/qwen3_*直接作为PowerShell路径，未展开通配而报123；改为features目录配-g qwen3_*.py，读取成功。ClearSight官方CVPR网页403，未声称读到；VISTA改读可用arXiv HTML方法。均未修改当前GPU运行或重试生成。

- 2026-11-20 region-commit启动：launcher猜测旧region100根有independent_summary.json，实际不存在，守卫在创建新GPU子进程前拒绝。只读核查旧analysis_complete为失败、analysis_recovery_complete为成功，且provenance／机制审核均passed；另存证书核原协议、恢复请求和10个caption SHA，启动器据真实恢复证据验证。保留launch_dependency_failure及旧失败记录，新方法／协议／控制输出未改，未重复原100。

- 2026-11-20 full-SADI-next6：两卡100图均生成rc0，机制审计通过；原分析误调region摘要，缺accepted字段而在指标落盘前失败。保留失败标记，独立恢复脚本固定caption SHA后直接使用支持原next统计的原evaluate_caption_chair.py；不改评分或重生成。恢复启动器首次用Paramiko open('x')只创建空文件但不能写入，父进程未启动；核该新文件为0字节后写入并正常启动CPU恢复。恢复请求及方法代码不变。

- 2026-11-20 signal-erasure：新probe的plan=dict尾部多一个右括号，local py_compile在上传／建根／GPU前拒绝。修正后重新编译、prepare和76项CPU数学检查通过才启动两卡原6／4组诊断；未覆盖冻结实验或重跑先前100图。

- 2026-11-20 preference-contrast审核：首次独立审核把torch.topk首项当作torch.argmax，在person／book分数并列的两项展示上失败。按实际生成规则，用保存的全部候选和FP32 logits作独立numpy.argmax，24个实际选择全匹配；两项并列另存sidecar，原始topk及生成未改。fork→nap的实际ID25859复算一致；不根据不稳定topk展示更换方法或声称效果。

- 2026-11-20 live-preference10：两卡生成与双评分成功，独立审计因协议漏写copied_controls元数据失败。冻结所有输出和协议，用独立recovery按四份对照字节收据在审计内存计划补该列表；数值／评分代码不改。首次直接SSH调用cwd为home，旧审计相对源码依赖找不到；从仓库cwd运行同一冻结脚本后60份输出和240配对区间全部通过，未重生成或重评分。未来100需在启动前显式验证copied_controls字段和审计cwd。

- 2026-11-20 preference100/excess闭合：CURRENT_TASK初稿8078B超过8000字节守卫，写入前拒绝，旧状态未改。缩短已归档旧节点细节后保存；未改任何冻结方法／指标或重启GPU。完整100与excess诊断本身无计算失败或重试。

- 2026-11-20 visualFFN10/dual闭合：临时状态更新命令 close_dual_attention_current.py 的草稿超过 CURRENT_TASK 8000 字节守卫，在写入前被拒绝，原状态未变。保留原状态归档，压缩已闭合 100/excess 历史后保存为 7401 字节；没有修改方法、协议、评分或重跑 GPU。两项新实验的 CPU/GPU/独立审计均无失败恢复。

- 2026-11-20 FFN/risk-plan：只读检索误用了不存在的 native_caption_source 快照、dgst_utils.py，及把 PowerShell 的 -ErrorAction 传给 rg，均改为实际 recovery/dgst_t 路径与目录 -g；Nature HB 正文访问被登录重定向拒绝，未声称读过。GPU 两项检查均返回 0。独立 risk-plan 审核误对 NumPy/Torch FP64 不同求和次序要求完全相等，在 table 的最终风险和相差 2.22e-16 处失败；保留独立失败侧录，聚合审核沿用同一脚本既有运输项 1e-12 容差，FP32 惩罚仍精确校验。六计划 HiGHS 与 network simplex 最大目标差 1.11e-16，原协议、指标、GPU 输出不改、不重跑。

- 2026-11-20 breadth32闭合：临时 risk_plan_breadth_close_current.py 的状态草稿超过8KB守卫，写入前拒绝，原状态未变。保留此前归档并缩短已闭合历史后保存7100B；未改被冻结的69位置协议、计算源、指标或GPU结果。两worker与207分支检查及独立69份LP审核全部通过，无模型恢复/重试。

- 2026-11-20 semantic-specificity 只读核查：误猜本机 CLVA 词表、region_rivals 目录，以及误以为旧协议包含 installed modeling 源码 SHA，均在读取／守卫处失败。改读 live_visual 的词表、supported_rivals 的实际产物，并用 sysconfig 确认 Python3.12 下的真实 transformers 模块。未改变模型、方法或旧结果。一次过宽的结构化打印包含 Q/P 长数组，后续只打印摘要。CURRENT_TASK 本轮初稿 7744B，守卫通过，未发生写入失败。

- 2026-11-20 decoupled-route：一次 functions.exec 编排缺闭括号，JavaScript 解析时拒绝，内部命令未执行；修正编排后运行。相对预算、早期缓存、分离目的地与首次 crop 检查均正常退出，未恢复模型或重跑描述。

- 2026-11-20 semantic-pair-FFN v1：GPU0 正常完成，GPU1 在验证 crop+route 旧基线时 TypeError。原因是旧结果字典 previous 被 teacher 输入列表同名覆盖；尚未完成辅助分支。保留 v1 源码、protocol、failure1、部分结果和日志，新增 v2 仅改字典名与恢复收据。GPU0 的三项无动作结果／worker／日志与 27 项 CPU 检查按字节 SHA 复用；只重算 GPU1，方法、参数、图像、裁剪及基线验证要求不变。旧对照输出未修改。

- 2026-11-20 live-semantic-crop10 启动前：只读命令误猜 run_qwen3_live_visual_ffn_contrast.py，改用 rg 找到实际 run_qwen3_live_visual_ffn_contrast10.py。CURRENT_TASK 增补草稿超过8KB守卫，在写前拒绝；已有状态归档保留，压缩闭合历史后写6835B。未改变任何实验或重跑 GPU。

- 2026-11-20 live-semantic-crop10 评分：双卡三策略完整生成均rc0，原句级评估因新候选行缺少 seconds 字段而KeyError，保留driver_failure与原始caption。独立 evaluate_qwen3_semantic_crop10_recovery.py 冻结原caption SHA，仅在派生评估输入加 seconds=batch_wall_seconds/batch_size；全部既有字段、文本与token IDs精确不变，对照按字节复制。用原冻结 sentence/legacy 评估器在新目录评分并复用可校验GT缓存；未改评估公式或重生成。

- 2026-11-20 live-semantic-crop10 评分恢复：v1 预建输出目录却未复制 evaluator.json，原评分器 provenance 守卫拒绝rc2；v2 连同 provenance 复制GT后，旧 summarize 又因候选 sar_ems_statistics 不含 guided_steps 而KeyError。两版源/请求/失败均保留。v3 新建派生输入，除 seconds 外将候选完整诊断字典改存 semantic_crop_statistics，反向还原逐行精确；先检查旧接口所需字段再评分。沿用未改动的原 sentence/legacy 评分器及相同GT缓存，两者均rc0；原caption SHA不变，GPU没有重跑。独立530项审核通过，三候选完整物体评分逐图全等SADI。
