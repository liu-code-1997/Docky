# M8-A 设计:多格式加载(PDF / HTML / docx / txt)

## 目标

让灌库能吃进 `.md` 以外的常见格式——PDF / HTML / docx / txt——把它们抽成纯文本后进入现有
切分→向量化链路。这是"通用小助手"的关键能力(换行业资料常是 PDF/网页/Word)。

**核心不变量**:`.md` 处理**完全不变**(现有 md-only 语料行为一致,回归安全);切分/向量化/检索链路
不动;只扩展"从文件取文本"这一层。

**依赖决策(已定,记录)**:多格式解析**必须引解析库**(纯 stdlib 做不好 PDF/docx)。选**轻量纯 Python**:
`pypdf`(PDF)、`beautifulsoup4`(HTML)、`python-docx`(docx);txt/md 用 stdlib。**拒绝重依赖**
(torch/onnx 等)。这与 M9"零依赖可行却引重库"性质不同——此处依赖是能力本身要求的。

## 1. 提取器注册表(新 `src/rag/extractors.py`)

按扩展名分发的纯函数。**解析库在函数内惰性 import**——导入 extractors 不强制装齐所有库,
且缺某库只影响该格式。

```
_extract_text(path): stdlib read_text            # .md / .txt
_extract_pdf(path):  pypdf.PdfReader → 各页 extract_text 拼接
_extract_html(path): bs4 去 script/style → get_text
_extract_docx(path): python-docx → 各段落 text 拼接
EXTRACTORS = {.md,.txt,.pdf,.html,.htm,.docx → fn}
extract_text(path) -> str    # 按后缀分发;不支持的后缀抛 ValueError
supported_suffixes() -> set
```

## 2. loader 分发(`src/rag/loader.py`)

`load_chunks_from_dir` 从"只扫 `*.md`"改为"扫所有受支持后缀的文件":

```
遍历 docs_dir 下所有 is_file 且后缀 ∈ supported_suffixes 的文件(排序,确定序)
  → extract_text(path) 取纯文本(空文本跳过)
  → 切分:.md 且 strategy==markdown → chunk_markdown(保持现状);其余 → chunk_text(char)
  → 组装 Chunk(id=source::i, source=相对路径, library=顶层目录)
```

**md 行为不变**:md 文件仍被扫到、仍用 read_text、strategy==markdown 时仍 chunk_markdown——
与改前逐字节等价。非 md 一律 char 切分(它们没有 `#` 结构)。`library`/`source` 规则不变。

## 3. 依赖

`pyproject.toml` dependencies 增加 `pypdf>=4`、`beautifulsoup4>=4.12`、`python-docx>=1.1`;
`.venv` 装上(需网络)。

## 4. 测试(TDD)

- `test_extractors`:txt/html/docx **可在测试内即时造文件**并验证提取(docx 用 python-docx 写一个再读;
  html 造带 script/style 的片段验证被剔除、正文保留;txt 直读)。**PDF 单测用一个极小提交的 fixture**
  `tests/fixtures/sample.pdf`(含已知文本)验证提取;若造 fixture 不便,则退化为集成验证(Task 见下)。
  extract_text 不支持后缀抛 ValueError;supported_suffixes 含全部。
- `test_loader`:混合格式目录(md + txt + html)→ 产出各格式的 chunk;**已有 md-only 用例仍通过**
  (回归:md 行为不变)。

## 5. 能力验证(退出条件)

放一个真实非 md 文件(如一小段 HTML 或 txt)进 `docs/<somelib>/`,重灌库,确认它变成可检索的 chunk
(count 增加、能被检索到)。这是**能力烟囱验证**,不改主评估集(56 题仍是 md 语料,用作回归——指标应不变)。

## 6. 影响文件
| 文件 | 改动 |
|---|---|
| `src/rag/extractors.py` | 新增:提取器 + 注册表 |
| `src/rag/loader.py` | 扫所有支持后缀 + 分发提取 |
| `pyproject.toml` | 加 pypdf/bs4/python-docx |
| `tests/test_extractors.py` | 新增 |
| `tests/test_loader.py` | 混合格式 + md 回归 |
| `notes/` | 简述多格式支持 |

## 7. 非目标(YAGNI)
- 不做 PDF 页码/HTML URL 等富元数据(Chunk 模型不加字段)。
- 不做格式感知结构切分(docx 标题/PDF 版面)——非 md 一律 char;结构化留后续。
- 不做 OCR(扫描件 PDF 无文本层则跳过)。
- 不动检索/生成/重排/embedding。
- 不为多格式扩主评估集(那是又一轮标注)。

## 8. 诚实提醒
- **引入 3 个运行时依赖**(pypdf/bs4/python-docx)——这是能力代价,已明确接受;都是轻量纯 Python。
- PDF 抽取质量看文档:纯文本层 PDF 好,扫描件/复杂版面可能抽出乱序或空(空则跳过,不崩)。
- md 语料的 56 题评估**指标应完全不变**(md 路径未改);变了就是回归,要查。
