# 当前 Qwen3 SVAR 对照的实现来源

重新核对了 [SVAR 作者实现](https://github.com/xjtu-wangzihan/SVAR/blob/main/modify_attention.py) 和 [使用说明](https://github.com/xjtu-wangzihan/SVAR/blob/main/README.md)。作者的操作是在 softmax 前，给末查询各视觉位置的每个头加上该位置各头 logit 绝对值均值的 alpha 倍。本项目保留这一操作，alpha=0.5，当前 Qwen3 实验固定在 0-based 第 5–18 层。与项目已有 SVAR 分支的精确一致性由 GPU 预检和前缀对齐核查。

当前对照是该操作在 Qwen3-VL 上的移植。作者仓库的模型加载器支持 LLaVA-1.5；其中的模型模板、视觉位置和运行环境不同于本项目 Qwen3 设置。因此不能将本项目数值表述为原论文 Qwen3 结果的复现。README 的命令范围、函数的半开区间和示例文件名存在端点表述差别；本项目协议明确保存实际生效的 5–18 层，不依赖命令文本推断。

2026-11-18 读取的 Git blob 标识：`modify_attention.py` 为 `c1d03f3b6eeb15ccd1f521472be78edfca38e65d`，`README.md` 为 `24e611d76a245b7be5a855fac6450b71e021c5c6`，`model_manager.py` 为 `be794c109138f951c3f8b87c0cc1b63af6728a50`。操作核对通过，没有发现需改变活跃实验实现的依据。
