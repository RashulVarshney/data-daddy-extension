import math

from eval.silver_queries import build_silver

from datadaddy_ai.retrieval.corpus import Card, chunk_card, strip_front_matter
from datadaddy_ai.retrieval.index import BM25Index, DenseIndex, HashEmbedder, HybridIndex, rrf, to_docs
from datadaddy_ai.retrieval.metrics import first_rank, hit_at_k, mrr_at_k, ndcg_at_k
from datadaddy_ai.retrieval.search import DatasetSearch


def mk(i, text, task="text-classification", lang="en", lic="mit", size="1K<n<10K"):
    return Card(f"org/ds{i}", text, [task], [lang], [lic], [size])


CARDS = [
    mk(0, "---\nlicense: mit\n---\n# Spam\nSMS messages labeled spam or ham.\n## Fields\nlabel and text"),
    mk(1, "# Recipes\nCooking recipes with ingredients and steps.", "text-generation"),
    mk(2, "# Tweets\nSentiment of tweets about airlines.", "text-classification", "en", "apache-2.0"),
    mk(3, "# Frases\nCorpus de traduccion espanol ingles.", "translation", "es", "cc-by-4.0", "100K<n<1M"),
    mk(4, "# Cars\nImages of car models for classification.", "image-classification"),
]


def test_front_matter_stripped():
    assert strip_front_matter("---\na: b\n---\n# T\nx").startswith("# T")


def test_chunking_by_section_with_cap_and_overlap():
    body = "# A\n" + " ".join(f"w{i}" for i in range(100)) + "\n## B\nshort"
    ch = chunk_card(mk(9, body), max_words=40, overlap=10)
    assert len(ch) >= 4 and all(c.text.startswith("Dataset org/ds9") for c in ch)
    assert any("short" in c.text for c in ch)
    sizes = [len(c.text.split("\n", 1)[1].split()) for c in ch]
    assert max(sizes) <= 40
    # overlap: consecutive windows of the long section share words
    w = [c.text.split("\n", 1)[1].split() for c in ch[:2]]
    assert set(w[0][-10:]) & set(w[1])


def test_bm25_finds_lexical_match():
    idx = BM25Index(["cooking recipes ingredients", "airline tweets sentiment", "car images"])
    assert idx.search("sentiment of airline tweets", 3)[0][0] == 1


def test_rrf_fuses_and_is_deterministic():
    a = [(1, 0.9), (2, 0.5), (3, 0.1)]
    b = [(3, 5.0), (1, 4.0), (9, 1.0)]
    f = rrf([a, b])
    assert f[0][0] == 1 and {i for i, _ in f} == {1, 2, 3, 9}
    assert rrf([a, b]) == f


def test_dense_and_hybrid_with_hash_embedder():
    texts = [c.text for c in CARDS]
    d = DenseIndex(HashEmbedder(), texts)
    assert d.search("sms spam ham", 3)[0][0] == 0
    h = HybridIndex(BM25Index(texts), d)
    assert h.search("spam sms labeled", 5)[0][0] == 0


def test_to_docs_dedupes_in_rank_order():
    assert to_docs([(0, 3.0), (1, 2.0), (2, 1.0)], ["a", "a", "b"], 5) == [("a", 3.0), ("b", 1.0)]


def test_metrics():
    ranked = ["x", "y", "z", "r"]
    rel = {"r"}
    assert first_rank(ranked, rel) == 4 and hit_at_k(ranked, rel, 3) == 0 and hit_at_k(ranked, rel, 5) == 1
    assert mrr_at_k(ranked, rel) == 0.25
    assert math.isclose(ndcg_at_k(["r", "a"], {"r"}), 1.0)
    assert 0 < ndcg_at_k(["a", "r"], {"r"}) < 1


def test_search_filters():
    s = DatasetSearch.hybrid(CARDS, HashEmbedder())
    r = s.search("dataset", 5, {"task": "translation"})
    assert [x["id"] for x in r] == ["org/ds3"]
    r = s.search("tweets airline", 3, {"license": "apache-2.0"})
    assert r[0]["id"] == "org/ds2"


def test_silver_generation_deterministic_and_consistent():
    cards = [
        mk(
            i,
            f"# d{i}\nthing {i}",
            ["text-classification", "translation"][i % 2],
            ["en", "fr", "de"][i % 3],
            ["mit", "apache-2.0"][i % 2],
            ["n<1K", "1K<n<10K", "10K<n<100K"][i % 3],
        )
        for i in range(60)
    ]
    a, b = build_silver(cards, 8, seed=1), build_silver(cards, 8, seed=1)
    assert a == b and len(a) == 8
    by = {c.id: c for c in cards}
    for q in a:
        for rid in q["relevant"]:
            c = by[rid]
            assert q["constraints"]["task"] in c.task_categories
