"""消融对比表格式化(M7)。把多组配置的 aggregate 结果排成一张 markdown 表。"""

# (aggregate 键, 表头显示名),顺序即列顺序
_COLUMNS = [
    ("hit_rate", "hit@k"),
    ("avg_recall", "recall@k"),
    ("avg_precision", "precision@k"),
    ("avg_ndcg", "nDCG@k"),
    ("avg_mrr", "MRR"),
    ("avg_gen_score", "gen"),
    ("refusal_accuracy", "拒答正确率"),
]


def format_comparison_table(results: list[dict]) -> str:
    """results: [{"config": str, "agg": {metric: value}}, ...] → markdown 表。"""
    header = "| 配置 | " + " | ".join(label for _, label in _COLUMNS) + " |"
    sep = "|" + "---|" * (len(_COLUMNS) + 1)
    lines = [header, sep]
    for r in results:
        agg = r["agg"]
        cells = " | ".join(f"{agg.get(key, 0.0):.3f}" for key, _ in _COLUMNS)
        lines.append(f"| {r['config']} | {cells} |")
    return "\n".join(lines)
