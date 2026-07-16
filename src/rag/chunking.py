"""文本切分。第一版:固定字符窗口 + 重叠。

坑(写进 notes/03):
- 切太大:检索不精准、塞爆 LLM;切太小:上下文断裂。
- 不理解语义边界,可能从句子/代码块中间切断。
- overlap 用于缓解"语义被切断",但会增加冗余存储。
"""
import re


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


# ---- M5 ①:markdown 结构切分 + 噪声过滤 ----

# 噪声段特征:命中任一即视为营销/导航噪声,丢弃(小写匹配)。
_NOISE_MARKERS = (
    "sponsor",
    "fastapi cloud",
    "conf",            # FastAPI Conf 会议海报
    "deploying to",    # 部署宣传样板
    "deployment successful",
)


def is_noise(heading: str, body: str) -> bool:
    """标题或正文命中噪声特征则判为噪声段。"""
    blob = f"{heading}\n{body}".lower()
    return any(marker in blob for marker in _NOISE_MARKERS)


def _strip_anchor(title: str) -> str:
    """去掉 mkdocs 标题里的锚点语法 { #some-id },它不是语义内容。"""
    return re.sub(r"\s*\{\s*#[^}]*\}\s*", "", title).strip()


def _is_html_boilerplate(body: str) -> bool:
    """判断一段是否几乎全是 HTML/样式标签、没有实质文字(应丢弃)。

    做法:先剔除 <style>/<script> 整块(内含 CSS/JS 不是正文),
    再按"HTML 标签密度"判断——纯样板块标签多、正文少;
    正常正文即便带少量行内 code/链接,标签也稀疏,不会误杀。
    """
    body = re.sub(r"<(style|script)[^>]*>.*?</\1>", "", body,
                  flags=re.DOTALL | re.IGNORECASE)  # 整块剔除样式/脚本
    if re.search(r"<(style|script)\b", body, re.IGNORECASE):
        return True  # 残缺未闭合的 style/script 起始标签,也当样板

    tags = re.findall(r"<[^>]+>", body)              # HTML 标签数
    imgs = re.findall(r"!\[[^\]]*\]\([^)]*\)", body)  # markdown 图片/徽章
    prose = re.sub(r"<[^>]*>", "", body)
    prose = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", prose)
    prose = re.sub(r"[\s\-_=*#/|>]", "", prose)       # 去空白与装饰符,剩实义字符

    if len(prose) < 5:
        return True                                   # 几乎没正文
    # 标签密度高(标签+图片数 相对 实义字符过多)= 样板
    return (len(tags) + len(imgs)) * 12 > len(prose)


def chunk_markdown(text: str, chunk_size: int, overlap: int) -> list[str]:
    """按 markdown 标题切分正文块,并过滤噪声/样板段。

    与 chunk_text(按字符硬切)相比,这里尊重文档结构:
    - 以 # / ## / ### ... 标题为边界切段,每段是语义完整的正文;
    - 命中噪声特征、或 HTML 标签密度过高的段直接丢弃;
    - 过长的段再退回字符窗口切分,避免塞爆。

    注意(M5 诊断结论):不把标题前缀拼进每块。文档标题常含高频词
    (如 "FastAPI"),前缀进正文会让门户页碎块与含该词的查询虚高相似度,
    挤掉真正相关的正文,反而拉低检索命中(实测 25% vs baseline 50%)。
    """
    lines = text.splitlines()
    heading_stack: list[tuple[int, str]] = []  # (level, title),仅用于噪声判定
    cur_heading = ""
    cur_body: list[str] = []
    chunks: list[str] = []

    def flush():
        body = "\n".join(cur_body).strip()
        if not body:
            return
        if is_noise(cur_heading, body) or _is_html_boilerplate(body):
            return
        # 不加标题前缀:正文若超长,按字符窗口切
        pieces = [body] if len(body) <= chunk_size else chunk_text(body, chunk_size, overlap)
        chunks.extend(pieces)

    for line in lines:
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            flush()
            cur_body = []
            level = len(m.group(1))
            title = _strip_anchor(m.group(2).strip())
            # 维护标题链(仅供 is_noise 判定用)
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            cur_heading = " > ".join(t for _, t in heading_stack)
        else:
            cur_body.append(line)
    flush()
    return chunks
