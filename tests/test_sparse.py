from rag.sparse import tokenize, encode_sparse


def test_tokenize_keeps_identifiers_lowercases():
    assert tokenize("Use item_id and LEFT JOIN") == ["use", "item_id", "and", "left", "join"]


def test_encode_sparse_counts_term_frequency():
    idx, val = encode_sparse("join join index")
    assert len(idx) == len(val) == 2            # 两个不同词
    assert sorted(val) == [1.0, 2.0]            # join=2, index=1


def test_encode_sparse_is_deterministic_across_calls():
    a = encode_sparse("query_points limit filter")
    b = encode_sparse("query_points limit filter")
    assert a == b                                # 稳定哈希:两次完全一致


def test_encode_sparse_empty():
    assert encode_sparse("") == ([], [])
