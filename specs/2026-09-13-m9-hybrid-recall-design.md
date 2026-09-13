# M9 设计:多路召回(dense + 稀疏 + RRF 融合)

## 目标

在现有稠密向量检索之外,加一条**稀疏(BM25 式词形)召回**,用 **RRF 融合**两路结果,
补稠密向量在**精确术语**上的短板(如 `item_id` / `LEFT JOIN` / `query_points`——中文问经改写
带出英文术语后,稀疏路能精确命中英文文档)。

**度量靶子**:M8 后的新基线(dense-only,rewrite 开,char,52 正例含 5 道多来源):
`hit@4 94.2% / recall@4 0.936 / precision@4 0.567 / nDCG@4 0.823 / MRR 0.796`。
M9 期望**抬升 recall@4**(hit 已近顶,靠多来源题的多命中覆盖体现价值)。

**核心决策(已定)**:
- **稀疏向量纯 Python 自算,零新依赖**(延续 tomllib 的零依赖作风)。否决 FastEmbed(重、拖 onnxruntime)。
- **Qdrant 原生命名向量(dense+sparse)+ `Fusion.RRF`**;`Modifier.IDF` 让服务端做 IDF 加权,我们只供词频。
- **spike 已验证**:qdrant-client 1.18 的 `:memory:` 本地模式支持命名 dense+sparse + RRF,故单测可用 `:memory:`。

## 1. 稀疏编码器(新 `src/rag/sparse.py`,纯函数零依赖)

```
tokenize: 小写 → 正则 [a-z0-9_]+ 取词(保留 item_id 这类标识符;中文不切,交给稠密路)
encode_sparse(text) -> (indices: list[int], values: list[float]):
    对每个 token 用 **稳定哈希** 映射到固定空间(2**20),值 = 词频(Counter)
```

**关键坑(诚实提醒)**:Python 内置 `hash()` 每进程随机(PYTHONHASHSEED),灌库进程和检索进程会
哈希出不同下标 → 完全对不上。**必须用稳定哈希**:`zlib.crc32(token.encode())`(stdlib,快,确定性)。
这是本里程碑最容易翻车的点,单测里显式验证跨调用确定性。

编码器返回 `(indices, values)` 元组(不依赖 qdrant),由 store 层包成 `models.SparseVector`——保持 sparse.py 可独立测试。

## 2. VectorStore 接口 + QdrantStore(命名向量)

现有是**未命名**稠密向量。M9 迁移到**命名向量** `dense` + `sparse`。这是新 collection schema →
**必须重灌库**(现有 144 块 unnamed 向量不兼容)。

接口(interfaces.py `VectorStore`)变更:
- `ensure_collection(vector_size)`:建 `vectors_config={"dense": VectorParams(size, COSINE)}` +
  `sparse_vectors_config={"sparse": SparseVectorParams(modifier=Modifier.IDF)}`。
- `upsert(chunks, vectors, sparse_vectors=None)`:sparse_vectors 为 `list[(indices,values)] | None`;
  point.vector = `{"dense": v, "sparse": SparseVector(...)}`(有 sparse 时)或 `{"dense": v}`。
- `search(query_vector, top_k, library=None)`:**保留**,内部查命名 `"dense"`(dense-only,行为等价现状)。
- **新增** `hybrid_search(query_vector, sparse_query, top_k, library=None)`:
  `query_points(prefetch=[Prefetch(dense, using="dense", limit=k*factor), Prefetch(sparse, using="sparse", limit=k*factor)], query=FusionQuery(RRF), limit=k, query_filter=...)`。
- `count` / `list_libraries` 不变。

**行为保持**:dense-only(hybrid 关)走命名 `"dense"`,同向量同 cosine → 检索结果与现状一致(只是 schema 命名化 + 需重灌)。

## 3. retrieve.py:混合路径

`retrieve(..., hybrid: bool = False)`:
- 改写得到 `query`;稠密 `embed_one(query_prefix + query)`;
- hybrid=False → `store.search(dense_vec, ...)`(现状)。
- hybrid=True → `sparse = encode_sparse(query)`(用改写后文本,**不加 embedding 前缀**——前缀只为稠密);
  `store.hybrid_search(dense_vec, sparse, ...)`。
- reranker(若开)在融合结果之后照旧执行。

## 4. config + 装配

- `Settings.hybrid: bool = False`(默认关=现状);`hybrid_prefetch_factor: int = 5`(每路召回 top_k×factor 再 RRF)。
- ingest.py:为每个 chunk 算 `encode_sparse(c.text)`(原文,不加 doc 前缀——前缀只为稠密),传给 upsert。
- pipeline.py:透传 `hybrid`。
- scripts(serve/eval/ingest/ask/ablate/gen_eval):从 settings 注入 hybrid;retrieve/pipeline 传 hybrid;ingest 传 sparse。

## 5. 度量(退出条件)

1. 迁移命名向量 + 稀疏,全部测试绿(dense-only 行为不变)。
2. **重灌一次**(存 dense+sparse)。然后**同一库**上 eval 对比 `hybrid 关 vs 开`(无需二次灌库,只切 retrieve 路径):
   记录 hit/recall/precision/nDCG/MRR。重点看 **recall@4 是否 > 0.936**。
3. 结论记入 notes/07。**有效则 `HYBRID=true` 设为推荐;无效则默认关**(机制保留,换语料可再验)。

## 6. 影响文件

| 文件 | 改动 |
|---|---|
| `src/rag/sparse.py` | 新增:tokenize + encode_sparse(crc32 稳定哈希) |
| `src/rag/interfaces.py` | VectorStore:upsert 加 sparse_vectors;新增 hybrid_search;ensure_collection 命名化 |
| `src/rag/providers/qdrant_store.py` | 命名向量 dense+sparse;upsert/search/hybrid_search 实现 |
| `src/rag/retrieve.py` | hybrid 路径 |
| `src/rag/ingest.py` | 算并存稀疏向量 |
| `src/rag/config.py` | hybrid + hybrid_prefetch_factor |
| `src/rag/pipeline.py` | 透传 hybrid |
| `scripts/*.py` | 装配注入 hybrid + sparse |
| `tests/*` | sparse 编码、命名向量 upsert/search/hybrid、retrieve 混合路径 |
| `notes/07` | 记录 hybrid on/off 对比 |

## 7. 测试(TDD)

- `test_sparse`:tokenize 保留 item_id;encode_sparse 词频正确;**跨调用确定性**(同文本两次结果一致——防 hash 随机)。
- `test_qdrant_store`(`:memory:`,spike 已证可行):命名向量 upsert;`search` dense-only 命中;`hybrid_search` RRF 融合返回两路都覆盖的点靠前;library 过滤仍生效。
- `test_retrieve`:hybrid=True 时调 hybrid_search 且稀疏取自改写后文本(用捕获 fake store)。
- `test_ingest`:upsert 收到 sparse_vectors(每 chunk 一个)。
- 现有 dense-only 测试:改造为命名向量后仍绿(行为等价)。

## 8. 非目标(YAGNI)

- 不引 FastEmbed / SPLADE / 外部稀疏模型。
- 不做可学习稀疏、不做查询词加权调参。
- 不做 DBSF(先只 RRF;RRF 是稳健默认)。
- 不动重排(M10)、生成(M11)。
- 稀疏分词只做英文/标识符;中文分词不做(交稠密路 + 查询改写英文术语覆盖)。

## 9. 诚实提醒

- **稳定哈希是硬约束**(见 §1),否则灌库/检索对不上,静默全崩。
- 命名向量是 schema 变更 → **必须重灌**;dense-only 也走命名 dense(等价但需重灌一次)。
- 稀疏路对**中文问/英文档**是否真帮上,取决于查询改写能否带出对的英文术语——**以 eval 数字为准**,无效就默认关(同 M8 前缀的教训)。
- 评估集仍 52 正例,多来源仅 5 道;recall 提升几个点要留意样本量,notes 如实标注。
