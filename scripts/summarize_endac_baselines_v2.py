#!/usr/bin/env python3
"""Write one readable four-model summary for the ENDAC baseline run."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "COCO4000-512-ENDAC-BASELINES-V2"
MODELS = (
    ("qwen2_5_vl_7b", "Qwen2.5-VL-7B"),
    ("llava_1_5_7b", "LLaVA-1.5-7B"),
    ("qwen3_vl_8b", "Qwen3-VL-8B"),
    ("internvl_2_5_8b", "InternVL2.5-8B"),
)


def _stat(row, *keys):
    value = row
    for key in keys:
        value = value[key]
    return float(value["mean"]), float(value["std"])


def _fmt(value: tuple[float, float]) -> str:
    return f"{value[0]:.4f} ± {value[1]:.4f}"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summaries = {}
    audits = {}
    for model, _display in MODELS:
        root = ROOT / "outputs" / model / "COCO4000-512-ENDAC-BASELINES-V2"
        result = root / "baseline" / "results" / f"{model}_baselines_3seed.json"
        audit = root / "baseline" / "merge_audit.json"
        if not result.exists():
            raise FileNotFoundError(result)
        if not audit.exists():
            raise FileNotFoundError(audit)
        summaries[model] = json.loads(result.read_text(encoding="utf-8"))
        audits[model] = json.loads(audit.read_text(encoding="utf-8"))

    lines = [
        "# COCO4000-512 ENDAC 独立 Baseline 完整结果",
        "",
        "六种 baseline 均只使用各自特征，没有与 risk、F 或 DGST 特征拼接。",
        "表中阈值由 train Real-F1 选择，测试集只做最终评估；数值为 seeds 43/44/45 的均值 ± 总体标准差。",
        "",
        "## 覆盖与标签",
        "",
        "| 模型 | ENDAC object tokens | Hall | Real | 六种 payload |",
        "|---|---:|---:|---:|---|",
    ]
    for model, display in MODELS:
        audit = audits[model]
        lines.append(
            f"| {display} | {audit['records']} | "
            f"{audit['labels']['hallucination']} | {audit['labels']['real']} | 完整 |"
        )

    auc_by_method: dict[str, list[float]] = {}
    for model, display in MODELS:
        summary = summaries[model]
        lines.extend(
            [
                "",
                f"## {display}",
                "",
                "| 方法 | Accuracy | Real F1 | Hall F1 | AUROC | Real AUPR | Hall AUPR |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        ranked = []
        for key, row in summary["methods"].items():
            metrics = row["test_metrics"]
            auc = _stat(metrics, "real_positive", "auc")
            ranked.append((auc[0], key, row, auc))
        for _auc_mean, key, row, auc in sorted(ranked, reverse=True):
            metrics = row["test_metrics"]
            display_name = row["display_name"]
            auc_by_method.setdefault(display_name, []).append(auc[0])
            lines.append(
                f"| {display_name} | {_fmt(_stat(metrics, 'accuracy'))} | "
                f"{_fmt(_stat(metrics, 'real_positive', 'f1'))} | "
                f"{_fmt(_stat(metrics, 'hallucination_positive', 'f1'))} | "
                f"{_fmt(auc)} | "
                f"{_fmt(_stat(metrics, 'real_positive', 'aupr'))} | "
                f"{_fmt(_stat(metrics, 'hallucination_positive', 'aupr'))} |"
            )

    lines.extend(
        [
            "",
            "## 四模型平均 AUROC",
            "",
            "| 方法 | 平均 AUROC | 覆盖模型数 |",
            "|---|---:|---:|",
        ]
    )
    for method, values in sorted(
        auc_by_method.items(), key=lambda item: mean(item[1]), reverse=True
    ):
        lines.append(f"| {method} | {mean(values):.4f} | {len(values)} |")

    lines.extend(
        [
            "",
            "## InsLen 设置",
            "",
            "- 固定 `K=16, m=4, scale=0.1 (temperature=10), bias=2, w=0.4`。",
            "- 仓库权重为 `0.6 × calibrated_local + 0.4 × context_consistency`。",
            "- LLaVA 使用仓库层位转换后的 `visual post-block=23, instruction post-block=21, object=final decoder norm`；其他骨干只对 visual/instruction 按相同相对深度映射，object 同样使用最终 decoder norm。",
            "- InsLen 是 training-free；表中的三 seed 数值相同，阈值仍只在 train split 上选择。",
            "",
            "## TokenGrounding 设置",
            "",
            "- 单一特征为 `concat(ads_per_layer, cgc_per_layer)`。",
            "- 独立一层 MLP：hidden width 248；不与用户方法拼接。",
        ]
    )
    path = OUTPUT / "complete_results.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
