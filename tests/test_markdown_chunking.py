from rag.chunking import chunk_markdown, is_noise


SAMPLE = """# FastAPI

FastAPI 是一个高性能的 Python Web 框架,基于类型注解构建 API。

## Path Parameters

路径参数在路径中用花括号声明,例如 /items/{item_id}。
该值会作为同名参数 item_id 传给你的函数。

## About FastAPI Cloud

FastAPI Cloud is the primary sponsor and funding provider for the project.
Deploying to FastAPI Cloud is easy.
"""


def test_chunk_markdown_splits_by_heading():
    chunks = chunk_markdown(SAMPLE, chunk_size=800, overlap=0)
    # 按标题切出独立正文块;正文本身在块内
    assert any("路径参数" in c and "花括号" in c for c in chunks)
    # 简介与 Path Parameters 是不同的块(按标题切开了)
    intro = [c for c in chunks if "高性能" in c]
    path = [c for c in chunks if "花括号" in c]
    assert intro and path and intro[0] != path[0]


def test_chunk_markdown_does_not_prefix_heading_into_body():
    # 诊断结论:把标题(常含高频词如 FastAPI)前缀进每块会污染检索相似度,故不再前缀。
    chunks = chunk_markdown(SAMPLE, chunk_size=800, overlap=0)
    target = [c for c in chunks if "花括号" in c][0]
    # 该正文块里不应混入其它小节标题当前缀
    assert not target.lstrip().startswith("Path Parameters")
    assert not target.lstrip().startswith("FastAPI")


def test_chunk_markdown_filters_noise_sections():
    chunks = chunk_markdown(SAMPLE, chunk_size=800, overlap=0)
    # 赞助/部署噪声段应被过滤掉
    assert not any("primary sponsor" in c for c in chunks)
    assert not any("Deploying to FastAPI Cloud" in c for c in chunks)


def test_chunk_markdown_long_section_still_windowed():
    text = "## Big\n\n" + ("句子。" * 1000)
    chunks = chunk_markdown(text, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    assert all(len(c) <= 800 for c in chunks)


def test_is_noise_flags_marketing_sections():
    assert is_noise("About FastAPI Cloud", "FastAPI Cloud is the primary sponsor") is True
    assert is_noise("FastAPI Conf 2026", "Join us in Amsterdam") is True
    assert is_noise("Path Parameters", "用花括号声明路径参数") is False


def test_strip_anchor_removes_mkdocs_anchor():
    from rag.chunking import _strip_anchor
    # mkdocs 标题锚点 { #id } 应被清掉(标题用于噪声判定,不该带锚点噪声)
    assert _strip_anchor("FastAPI { #fastapi }") == "FastAPI"
    assert _strip_anchor("Path Parameters { #path-parameters }") == "Path Parameters"
    assert _strip_anchor("No Anchor Here") == "No Anchor Here"


def test_chunk_markdown_drops_html_boilerplate_blocks():
    text = (
        "# FastAPI { #fastapi }\n\n"
        '<style>\n.md h1 { display: none }\n</style>\n\n'
        "## Real Section\n\n这是一段真正的正文,讲解路径参数用法。\n"
    )
    chunks = chunk_markdown(text, 800, 0)
    # 纯 HTML/样式样板块(无实质文字)应被丢弃
    assert not any("<style>" in c for c in chunks)
    assert not any("display: none" in c for c in chunks)
    # 真正的正文块保留
    assert any("真正的正文" in c for c in chunks)


def test_chunk_markdown_drops_high_tag_density_blocks():
    # 夹着几个词的 HTML 块(证言/徽章/图片),标签密度高,应丢弃
    text = (
        "## Opinions\n\n"
        '<div class="panel"><a href="x"><strong>Netflix</strong></a> is pleased '
        'to announce</div>\n<img src="badge.png">\n'
        "## Path Parameters\n\n路径参数用花括号声明,这是完整的一段中文正文说明。\n"
    )
    chunks = chunk_markdown(text, 800, 0)
    assert not any("Netflix" in c for c in chunks)      # 高标签密度块丢弃
    assert any("路径参数" in c for c in chunks)          # 正文保留


def test_chunk_markdown_keeps_prose_with_inline_code_and_links():
    # 正常正文里带少量行内 code/链接,不应被误杀
    text = "## Body\n\n用 `Pydantic` 的 [BaseModel](https://x) 定义请求体,声明字段类型即可完成校验。\n"
    chunks = chunk_markdown(text, 800, 0)
    assert any("Pydantic" in c for c in chunks)
