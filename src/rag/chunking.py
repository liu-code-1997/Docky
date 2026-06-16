"""文本切分。第一版:固定字符窗口 + 重叠。

坑(写进 notes/03):
- 切太大:检索不精准、塞爆 LLM;切太小:上下文断裂。
- 不理解语义边界,可能从句子/代码块中间切断。
- overlap 用于缓解"语义被切断",但会增加冗余存储。
"""


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += step
    return chunks
