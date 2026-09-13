# M8-A 多格式加载 Implementation Plan

> REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox steps.

**Goal:** 让灌库支持 PDF/HTML/docx/txt(抽纯文本→现有切分链路),md 行为不变。

**Spec:** `specs/2026-09-13-m8a-multiformat-design.md`

## Global Constraints
- `.venv/bin/python -m pytest`(venv=3.12;系统 3.9 会挂)。
- 新增 3 个轻量依赖(pypdf/beautifulsoup4/python-docx)——能力所需,已决策接受;拒绝重依赖。
- md 路径逐字节不变(回归安全);切分/向量化/检索不动。conventional commits。

---

### Task 1: 依赖 + extractors.py 提取器注册表

**Files:** Modify `pyproject.toml`; Create `src/rag/extractors.py`, `tests/test_extractors.py`

**Interfaces:** `extract_text(path)->str`(按后缀分发,不支持抛 ValueError);`supported_suffixes()->set`;`EXTRACTORS` dict。

- [ ] **Step 1: 装依赖** 在 `pyproject.toml` 的 `dependencies` 加三行:
```
    "pypdf>=4",                 # PDF 抽取
    "beautifulsoup4>=4.12",     # HTML 抽取
    "python-docx>=1.1",         # docx 抽取
```
装到 venv(需网络):`cd /Users/mi/Documents/test/rag-docs && .venv/bin/pip install "pypdf>=4" "beautifulsoup4>=4.12" "python-docx>=1.1"`

- [ ] **Step 2: 写失败测试** `tests/test_extractors.py`:

```python
import pytest
from rag.extractors import extract_text, supported_suffixes


def test_txt_and_md_extract_plain(tmp_path):
    (tmp_path / "a.txt").write_text("hello txt", encoding="utf-8")
    assert extract_text(tmp_path / "a.txt") == "hello txt"
    (tmp_path / "b.md").write_text("# T\nbody here", encoding="utf-8")
    assert "body here" in extract_text(tmp_path / "b.md")


def test_html_strips_tags_and_scripts(tmp_path):
    (tmp_path / "a.html").write_text(
        "<html><head><style>x{color:red}</style>"
        "<script>var y=1</script></head>"
        "<body><h1>Title</h1><p>Real content</p></body></html>",
        encoding="utf-8")
    out = extract_text(tmp_path / "a.html")
    assert "Real content" in out and "Title" in out
    assert "var y" not in out and "color:red" not in out


def test_docx_extracts_paragraphs(tmp_path):
    from docx import Document
    d = Document(); d.add_paragraph("First para"); d.add_paragraph("Second para")
    d.save(str(tmp_path / "a.docx"))
    out = extract_text(tmp_path / "a.docx")
    assert "First para" in out and "Second para" in out


def test_pdf_is_supported():
    assert ".pdf" in supported_suffixes()


def test_unsupported_suffix_raises(tmp_path):
    (tmp_path / "a.xyz").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        extract_text(tmp_path / "a.xyz")
```

- [ ] **Step 3: 确认失败** `.venv/bin/python -m pytest tests/test_extractors.py -v` → FAIL

- [ ] **Step 4: 实现** `src/rag/extractors.py`(解析库惰性 import):

```python
"""按扩展名分发的文本提取器。解析库在函数内惰性 import:导入本模块不强制装齐所有库,
缺某库只影响该格式。"""
from pathlib import Path


def _extract_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def _extract_html(path: Path) -> str:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    return soup.get_text(separator="\n")


def _extract_docx(path: Path) -> str:
    from docx import Document
    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs)


EXTRACTORS = {
    ".md": _extract_text, ".txt": _extract_text,
    ".pdf": _extract_pdf,
    ".html": _extract_html, ".htm": _extract_html,
    ".docx": _extract_docx,
}


def supported_suffixes() -> set[str]:
    return set(EXTRACTORS)


def extract_text(path: Path) -> str:
    fn = EXTRACTORS.get(Path(path).suffix.lower())
    if fn is None:
        raise ValueError(f"不支持的格式: {Path(path).suffix}")
    return fn(Path(path))
```

- [ ] **Step 5: 确认通过** `.venv/bin/python -m pytest tests/test_extractors.py -v` → PASS;全量 `.venv/bin/python -m pytest -q` 绿。
- [ ] **Step 6: 提交** `git add pyproject.toml src/rag/extractors.py tests/test_extractors.py && git commit -m "feat(M8-A): 多格式文本提取器(pdf/html/docx/txt)+ 轻量解析依赖"`

---

### Task 2: loader 按后缀分发

**Files:** Modify `src/rag/loader.py`; Test `tests/test_loader.py`

**Interfaces:** `load_chunks_from_dir` 扫所有受支持后缀;md+markdown 策略保持,其余 char。签名不变。

- [ ] **Step 1: 写失败测试** 在 `tests/test_loader.py` 追加(保留现有 md 用例):

```python
def test_loader_handles_mixed_formats(tmp_path):
    from rag.loader import load_chunks_from_dir
    lib = tmp_path / "lib"; lib.mkdir()
    (lib / "a.md").write_text("# H\nmarkdown body", encoding="utf-8")
    (lib / "b.txt").write_text("plain text body", encoding="utf-8")
    (lib / "c.html").write_text("<body><p>html body content</p></body>", encoding="utf-8")
    chunks = load_chunks_from_dir(tmp_path, chunk_size=800, overlap=0)
    srcs = {c.source for c in chunks}
    assert {"lib/a.md", "lib/b.txt", "lib/c.html"} <= srcs
    assert all(c.library == "lib" for c in chunks)
    assert any("html body content" in c.text for c in chunks)
```

- [ ] **Step 2: 确认失败** `.venv/bin/python -m pytest tests/test_loader.py -k mixed -v` → FAIL(现在只扫 .md)

- [ ] **Step 3: 实现** 把 `src/rag/loader.py` 的 `load_chunks_from_dir` 改为:

```python
from pathlib import Path
from rag.models import Chunk
from rag.chunking import chunk_text, chunk_markdown, _DEFAULT_NOISE_MARKERS
from rag.extractors import extract_text, supported_suffixes


def load_chunks_from_dir(docs_dir: Path, chunk_size: int, overlap: int,
                         strategy: str = "char",
                         noise_markers: tuple[str, ...] | list[str] = _DEFAULT_NOISE_MARKERS) -> list[Chunk]:
    docs_dir = Path(docs_dir)
    supported = supported_suffixes()
    chunks: list[Chunk] = []

    paths = sorted(p for p in docs_dir.rglob("*")
                   if p.is_file() and p.suffix.lower() in supported)
    for path in paths:
        rel = path.relative_to(docs_dir)
        library = rel.parts[0] if len(rel.parts) > 1 else "root"
        source = rel.as_posix()

        text = extract_text(path)
        if not text.strip():
            continue
        if path.suffix.lower() == ".md" and strategy == "markdown":
            pieces = chunk_markdown(text, chunk_size=chunk_size, overlap=overlap,
                                    noise_markers=noise_markers)
        else:
            pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            chunks.append(Chunk(id=f"{source}::{i}", text=piece,
                                source=source, library=library, chunk_index=i))
    return chunks
```

- [ ] **Step 4: 确认通过** `.venv/bin/python -m pytest tests/test_loader.py -v` → PASS(含现有 md 回归用例);全量 `.venv/bin/python -m pytest -q` 绿。
- [ ] **Step 5: 提交** `git add src/rag/loader.py tests/test_loader.py && git commit -m "feat(M8-A): loader 按后缀分发多格式(md 行为不变)"`

---

### Task 3: 能力烟囱验证(实机,控制器执行)

- [ ] **Step 1: 造一个非 md 样本** 在 `docs/` 下放一个真实非 md 文件(如 `docs/qdrant/extra.txt` 写一小段可检索文本,或一个小 HTML)。
- [ ] **Step 2: 重灌** `.venv/bin/python scripts/ingest.py` → 确认 count 较之前增加(非 md 被吃进)。
- [ ] **Step 3: 检索验证** 问一个只该样本能答的问题(scripts/ask.py 或直接 retrieve),确认命中该文件。
- [ ] **Step 4: md 回归** 跑 eval(56 题),确认指标与之前一致(md 路径未变);记一句到 notes。清理临时样本或保留(自定)。

---

## Self-Review
- Spec §1 extractors→T1;§2 loader→T2;§3 依赖→T1;§4 测试→T1/T2;§5 验证→T3;§6 文件全覆盖;§7 非目标未越界(不做富元数据/结构切分/OCR)。
- md 回归:T2 实现里 md+markdown 仍走 chunk_markdown、md 仍被扫到、extract_text(.md)=read_text,与改前等价;现有 test_loader md 用例保留即回归网。
- PDF 单测降级为"注册可用"(§4)+ T3 集成(造 fixture PDF 不便,不阻塞);txt/html/docx 全 hermetic。
- 类型:extract_text/supported_suffixes 在 T1 定,T2 loader 消费一致。
