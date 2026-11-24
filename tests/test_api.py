import pytest
from fastapi.testclient import TestClient

from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:   # runs lifespan -> loads models
        yield c


def post(client, **kw):
    base = {"user_id": 42, "amount": 1500, "card_type": "VISA", "merchant_category": "grocery",
            "location": "Mumbai", "device_type": "mobile_app", "timestamp": "2026-03-06T11:00:00"}
    return client.post("/score_transaction", json={**base, **kw})


def test_health(client):
    assert client.get("/health").json()["models_loaded"] is True


def test_normal_transaction_is_approved(client):
    r = post(client)
    assert r.status_code == 200
    assert r.json()["decision"] == "APPROVE"


def test_fraud_like_transaction_is_blocked(client):
    r = post(client, amount=95000, device_type="pos_terminal", location="Chennai")
    body = r.json()
    assert r.status_code == 200
    assert body["decision"] in {"VERIFY", "BLOCK"}
    assert body["risk_score"] > 30 and body["reasons"]


def test_score_is_monotonic_in_amount(client):
    low = post(client, amount=500).json()["risk_score"]
    high = post(client, amount=95000).json()["risk_score"]
    assert high > low


def test_bad_types_return_422(client):
    assert post(client, amount="abc").status_code == 422
    assert post(client, amount=-5).status_code == 422
    assert post(client, card_type="AMEX").status_code == 422
    assert post(client, merchant_category="not_a_category").status_code == 422


def test_unknown_user_cold_start(client):
    r = post(client, user_id=999999)
    assert r.status_code == 200 and r.json()["known_user"] is False


def test_audit_log_written(client):
    post(client)
    assert len(client.get("/audit_log?limit=5").json()) >= 1
