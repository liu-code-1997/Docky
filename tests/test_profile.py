from pathlib import Path

import pytest

from rag.profile import DomainProfile, load_profile


def test_load_tech_profile():
    p = load_profile("tech")
    assert "技术文档" in p.persona
    assert p.refusal_text == "根据现有资料无法回答"
    assert p.refusal_marker == "无法回答"
    assert "{question}" in p.rewrite_prompt          # 改写 prompt 保留占位符
    assert "sponsor" in p.noise_markers
    # Task 1–8 期间 tech 前缀留空(Task 9 才启用)
    assert p.embed_query_prefix == ""
    assert p.embed_doc_prefix == ""


def test_load_generic_profile():
    p = load_profile("generic")
    assert p.rewrite_prompt == ""                    # generic 不改写
    assert p.noise_markers == []


def test_missing_profile_raises():
    with pytest.raises(FileNotFoundError):
        load_profile("no_such_profile")


def test_defaults_fill_missing_fields(tmp_path):
    (tmp_path / "min.toml").write_text('persona = "只有人设"\n', encoding="utf-8")
    p = load_profile("min", profiles_dir=tmp_path)
    assert p.persona == "只有人设"
    assert p.refusal_marker == "无法回答"             # 缺字段回落默认
