"""领域 profile:把技术文档专属的耦合点(人设/改写/噪声/前缀)外置成可切换配置。

用 stdlib tomllib 读 profiles/<name>.toml(零依赖)。字段均有默认,缺字段回落。
"""
import tomllib
from pathlib import Path

from pydantic import BaseModel


class DomainProfile(BaseModel):
    persona: str = "你是一个严谨的问答助手"
    refusal_text: str = "根据现有资料无法回答"
    refusal_marker: str = "无法回答"
    rewrite_prompt: str = ""                 # 空 = 不改写
    noise_markers: list[str] = []
    embed_query_prefix: str = ""
    embed_doc_prefix: str = ""


def load_profile(name: str, profiles_dir: Path = Path("profiles")) -> DomainProfile:
    path = Path(profiles_dir) / f"{name}.toml"
    if not path.exists():
        raise FileNotFoundError(f"profile 不存在: {path}")
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return DomainProfile(**data)
