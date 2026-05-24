from db.bm25 import BM25TaskSearch, rrf_fusion, tokenize


def test_tokenize_lowercases_words():
    assert tokenize("Fix AUTH bug") == ["fix", "auth", "bug"]


def test_tokenize_supports_cyrillic_words():
    assert tokenize("Исправить авторизацию") == ["исправить", "авторизацию"]


def test_bm25_search_returns_best_matching_document():
    search = BM25TaskSearch(
        ids=["a", "b"],
        documents=["fix auth token", "render dashboard chart"],
        metadatas=[
            {
                "created_at": "2024-05-06T10:00:00+03:00",
                "finished_at": "2024-05-06T18:00:00+03:00",
            },
            {
                "created_at": "2024-05-06T10:00:00+03:00",
                "finished_at": "2024-05-06T18:00:00+03:00",
            },
        ],
    )

    results = search.search("auth token", n_results=1)

    assert results[0]["id"] == "a"
    assert isinstance(results[0]["bm25_score"], float)


def test_bm25_search_filters_by_business_days():
    search = BM25TaskSearch(
        ids=["fast", "slow"],
        documents=["fix bug", "fix bug"],
        metadatas=[
            {
                "created_at": "2024-05-06T10:00:00+03:00",
                "finished_at": "2024-05-06T18:00:00+03:00",
            },
            {
                "created_at": "2024-05-06T10:00:00+03:00",
                "finished_at": "2024-05-17T18:00:00+03:00",
            },
        ],
    )

    results = search.search("fix", n_results=10, where_days=(5, 20))

    assert [item["id"] for item in results] == ["slow"]


def test_rrf_fusion_combines_vector_and_bm25_results():
    vector_results = {"ids": [["a", "b"]]}
    bm25_results = [{"id": "b"}, {"id": "c"}]

    results = rrf_fusion(vector_results, bm25_results, top_n=3)

    assert [item["id"] for item in results] == ["b", "a", "c"]
    assert all("hybrid_score" in item for item in results)


def test_rrf_fusion_handles_empty_vector_results():
    vector_results = {"ids": [[]]}
    bm25_results = [{"id": "a"}]

    results = rrf_fusion(vector_results, bm25_results, top_n=3)

    assert results[0]["id"] == "a"
