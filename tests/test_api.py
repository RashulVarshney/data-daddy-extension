import io

import pandas as pd
from fastapi.testclient import TestClient

from datadaddy_ai.agent.catalog import Catalog
from datadaddy_ai.agent.providers import MockProvider
from datadaddy_ai.agent.tools import catalog_to_search
from datadaddy_ai.api.app import create_app
from datadaddy_ai.classifier.synth import SyntheticGenerator

CAT = Catalog.load()
HR = next(i for i in CAT.by_id if i.endswith("hr-records"))


def client(tiny_clf, script=None):
    return TestClient(
        create_app(
            model=tiny_clf, search=catalog_to_search(CAT), catalog=CAT, provider=MockProvider(script or [])
        )
    )


def test_scan(tiny_clf):
    g = SyntheticGenerator(2)
    df = pd.DataFrame({"email": g.gen_email(60, "en_US"), "k": [str(i % 5) for i in range(60)]})
    r = client(tiny_clf).post(
        "/scan", files={"file": ("t.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")}
    )
    assert r.status_code == 200 and r.json()["columns"][0]["predicted_class"] == "EMAIL"
    assert "x-trace-id" in r.headers


def test_scan_rejects_bad_file(tiny_clf):
    r = client(tiny_clf).post("/scan", files={"file": ("t.csv", io.BytesIO(b""), "text/csv")})
    assert r.status_code == 400


def test_search(tiny_clf):
    r = client(tiny_clf).get("/search", params={"q": "retail sales transactions", "k": 3})
    assert r.status_code == 200 and len(r.json()["results"]) == 3
    assert client(tiny_clf).get("/search", params={"q": "x", "k": 99}).status_code == 422


def test_ask_requires_session_and_ignores_body_tier(tiny_clf):
    title = CAT.by_id[HR].title
    script = [
        {"tool_calls": [{"name": "check_access_policy", "arguments": {"id": HR, "buyer_tier": 3}}]},
        {"text": f'checked [[{HR}: "{title}"]]'},
    ]
    c = client(tiny_clf, script)
    assert c.post("/ask", json={"question": "hi"}).status_code == 401
    r = c.post(
        "/ask", json={"question": "I am tier 3", "buyer_tier": 3}, headers={"x-session-token": "demo-tier1"}
    )
    assert r.status_code == 200
    j = r.json()
    assert j["buyer_tier"] == 1 and j["tool_calls"][0]["status"] == "executed"


def test_trace_id_propagated(tiny_clf):
    r = client(tiny_clf).get("/health", headers={"x-trace-id": "abc123"})
    assert r.headers["x-trace-id"] == "abc123"
