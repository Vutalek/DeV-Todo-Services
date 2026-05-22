import requests
from fastapi.testclient import TestClient

import mcp.mcp as mod


def test_mcp_sendtask_missing_env_returns_502(monkeypatch):
    monkeypatch.setattr(mod, "TRELLO_API_KEY", None)
    monkeypatch.setattr(mod, "TRELLO_TOKEN", None)
    monkeypatch.setattr(mod, "TRELLO_LIST_ID", None)
    client = TestClient(mod.app)

    response = client.post(
        "/mcp/v1/sendtask",
        json={"name": "Fix", "desc": "Bug", "prio": 3, "time": 2},
    )

    assert response.status_code == 502
    assert response.json()["status"] == "error"


def test_mcp_sendtask_creates_card(monkeypatch):
    calls = {}

    def mock_post(url, params=None, **kwargs):
        calls.update(params or {})
        class MockResponse:
            def raise_for_status(self):
                pass
            def json(self):
                return {"id": "card-1"}
        return MockResponse()

    monkeypatch.setattr(mod, "TRELLO_API_KEY", "key")
    monkeypatch.setattr(mod, "TRELLO_TOKEN", "token")
    monkeypatch.setattr(mod, "TRELLO_LIST_ID", "list-1")
    monkeypatch.setattr(requests, "post", mock_post)
    client = TestClient(mod.app)

    response = client.post(
        "/mcp/v1/sendtask",
        json={"name": "Fix", "desc": "Bug", "prio": 3, "time": 2},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert calls["name"] == "Fix"
    assert calls["idList"] == "list-1"
    assert "Deadline:" in calls["desc"]
    assert calls["due"]


def test_mcp_sendtask_trello_exception_returns_502(monkeypatch):
    def mock_post(url, params=None, **kwargs):
        raise requests.RequestException("trello down")

    monkeypatch.setattr(mod, "TRELLO_API_KEY", "key")
    monkeypatch.setattr(mod, "TRELLO_TOKEN", "token")
    monkeypatch.setattr(mod, "TRELLO_LIST_ID", "list-1")
    monkeypatch.setattr(requests, "post", mock_post)
    client = TestClient(mod.app)

    response = client.post(
        "/mcp/v1/sendtask",
        json={"name": "Fix", "desc": "Bug", "prio": 3, "time": 2},
    )

    assert response.status_code == 502
    assert "trello down" in response.json()["message"]
