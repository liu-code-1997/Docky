"""稀疏(BM25 式词形)向量编码 —— 纯 Python 零依赖。

用 zlib.crc32 稳定哈希(禁用内置 hash():它每进程随机,会让灌库与检索对不上)。
值=词频,IDF 交给 Qdrant 服务端的 Modifier.IDF。
"""
import re
import zlib
from collections import Counter

_TOKEN_RE = re.compile(r"[a-z0-9_]+")
_SPACE = 2 ** 20


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def encode_sparse(text: str) -> tuple[list[int], list[float]]:
    counts: Counter[int] = Counter()
    for tok in tokenize(text):
        counts[zlib.crc32(tok.encode("utf-8")) % _SPACE] += 1
    indices = list(counts.keys())
    values = [float(counts[i]) for i in indices]
    return indices, values
