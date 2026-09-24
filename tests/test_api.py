import os
import tempfile

os.environ["GEN4_DB"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["GEN4_LLM_PROVIDER"] = "offline"

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

c = TestClient(app)


def test_full_loop():
    c.put("/suites/geo", json=[{"id": "g1", "question": "Capital of India?", "expected_answer": "New Delhi"}])
    r = c.post("/runs", json={"suite": "geo", "model_label": "m1", "runs": 2,
                              "responses": {"g1": ["Delhi", "New Delhi"]}}).json()
    assert r["accuracy"] == 0.5
    c.post("/adjudicate", json={"suite": "geo", "question_id": "g1", "answer": "Delhi", "correct": True})
    assert c.post(f"/runs/{r['run_id']}/regrade").json()["after"] == 1.0
    assert c.post(f"/suites/geo/baseline/{r['run_id']}").status_code == 200
    assert c.get(f"/runs/{r['run_id']}").json()["model_label"] == "m1"


def test_errors():
    assert c.post("/runs", json={"suite": "nope", "model_label": "m"}).status_code == 404
    assert c.get("/runs/nope").status_code == 404
    assert "sample" in c.get("/suites").json()
