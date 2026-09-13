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
