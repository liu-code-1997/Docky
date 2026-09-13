"""缓解 lost-in-the-middle:把最相关的块放到上下文两端(M11)。"""
from rag.models import RetrievedChunk


def reorder_for_long_context(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """输入按相关性降序。最相关放首、次相关放尾,由外向内交替填充。"""
    n = len(chunks)
    if n <= 2:
        return list(chunks)
    result: list[RetrievedChunk | None] = [None] * n
    left, right = 0, n - 1
    for i, rc in enumerate(chunks):        # i = 相关性排名(0 最高)
        if i % 2 == 0:
            result[left] = rc; left += 1
        else:
            result[right] = rc; right -= 1
    return [rc for rc in result if rc is not None]
