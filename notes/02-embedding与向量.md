# 02 · Embedding 与向量

## 什么是 embedding(嵌入)

**Embedding = 把一段文本变成一串数字(向量)。** 这串数字"代表"了这段文本的语义。

例如 `nomic-embed-text` 把任意文本变成一个 **768 个浮点数**的向量:
```
"hello"  →  [0.42, -0.13, -4.12, ... ]   (共 768 个数)
```

为什么要这么做?因为计算机不擅长直接比较"两段话意思像不像",但**很擅长比较两串数字的距离**。embedding 的魔力在于:**语义相近的文本,它们的向量在 768 维空间里也"挨得近";语义无关的,离得远。**

- "如何定义路径操作" 和 "怎么写一个 API 路由" → 向量接近
- "如何定义路径操作" 和 "今天午饭吃什么" → 向量很远

## 怎么衡量"近"——余弦相似度

最常用的是**余弦相似度**(cosine similarity):看两个向量的"方向"有多一致,而不在乎长度。

- 方向完全一致 → 相似度 = 1(最相似)
- 互相垂直 → 相似度 = 0(无关)
- 方向相反 → 相似度 = -1

我们建 Qdrant 集合时指定的就是 `Distance.COSINE`,检索时返回的 `score` 就是这个相似度。验收时看到集合信息里 `距离=Cosine` 即此。

## 本项目的实际数据

- 模型:`nomic-embed-text`,维度 **768**。
- 这个维度**必须在建集合前就知道**,因为 Qdrant 集合要预先声明向量维度。
- 我们的做法(`scripts/ingest.py`):先用一句 `"probe"` 调用一次 embedding,拿到 `len(vector)=768`,再用这个维度建集合。这样**换 embedding 模型时维度自动适配**,不用写死。

```python
probe = embedder.embed_one("probe")
vector_size = len(probe)          # 768
store.ensure_collection(vector_size=vector_size)
```

## 坑:embedding 有最大输入长度,过长会被静默截断

每个 embedding 模型都有最大能处理的输入长度(token 数)。如果你把一大段文本(比如没切分的整篇文档)直接丢进去:

- 模型可能**只编码前面一部分,后面被悄悄丢掉**,而且**不报错**。
- 结果:你以为存进去的是整段内容,其实向量只代表了开头一部分,检索质量受损却毫无察觉。

这正是**为什么必须先切分(chunking)再 embedding** —— 见 [[03-切分chunking]]。把文本切成合适大小的块,既避免截断,也让检索更精准。

## 我们的 Embedder 是"可插拔"的

`OllamaEmbedder` 实现了 `Embedder` 抽象接口(`embed` / `embed_one`)。
以后切到云端 API(如 OpenAI 的 embedding),只要写一个同样实现 `Embedder` 的新类,核心逻辑(ingest/检索)一行都不用改。详见 [[07-可插拔设计]]。

---
相关笔记:[[00-RAG是什么]] · [[03-切分chunking]] · [[04-检索retrieval]] · [[07-可插拔设计]]
