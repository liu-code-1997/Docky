from rag.ablate import format_comparison_table


def test_format_comparison_table_markdown():
    results = [
        {"config": "rewrite=off rerank=off",
         "agg": {"hit_rate": 0.4, "avg_recall": 0.3, "avg_precision": 0.1,
                 "avg_ndcg": 0.35, "avg_mrr": 0.3, "avg_gen_score": 0.6,
                 "refusal_accuracy": 1.0}},
        {"config": "rewrite=on rerank=on",
         "agg": {"hit_rate": 0.9, "avg_recall": 0.8, "avg_precision": 0.4,
                 "avg_ndcg": 0.85, "avg_mrr": 0.75, "avg_gen_score": 0.82,
                 "refusal_accuracy": 1.0}},
    ]
    table = format_comparison_table(results)
    lines = table.splitlines()
    assert lines[0].startswith("| 配置 |")
    assert set(lines[1]) <= set("|-")          # 分隔行只有 | 和 -
    assert "rewrite=off rerank=off" in table
    assert "0.900" in table                     # hit_rate=0.9 → 0.900
    assert len(lines) == 4                       # 表头 + 分隔 + 2 行数据
