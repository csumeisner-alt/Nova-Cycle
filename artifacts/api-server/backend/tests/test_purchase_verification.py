"""
Tests for server-side Google Play purchase verification (Mint Luxe).

Uses a fake Play verifier via FastAPI dependency override so no real
Play Developer API calls happen. Covers:
  - valid purchase → entitled + entitlement row persisted
  - fake/nonexistent token → entitled false, nothing persisted
  - refund detected on re-check → entitlement revoked
  - verifier not configured → 503 (explicit, not silent)
  - Play API down → 502
  - pending purchase → not entitled
"""

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.db import get_session
from database.models import Base, PurchaseEntitlement
from billing.play_verifier import (
    InvalidPurchaseToken,
    PlayApiError,
    PlayPurchase,
    VerifierNotConfigured,
    get_play_verifier,
)
from main import app

PRODUCT = "mint_luxe_theme"


class FakeVerifier:
    """Scriptable stand-in for PlayPurchaseVerifier."""

    def __init__(self):
        # token -> PlayPurchase | Exception
        self.responses = {}

    async def verify(self, product_id, purchase_token):
        outcome = self.responses.get(purchase_token)
        if outcome is None:
            raise InvalidPurchaseToken("unknown token")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest_asyncio.fixture
async def client_and_deps(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async def override_get_session():
        async with session_factory() as session:
            yield session

    fake = FakeVerifier()
    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_play_verifier] = lambda: fake

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, fake, session_factory

    app.dependency_overrides.clear()
    await engine.dispose()


def _purchased(order_id="GPA.1234-5678"):
    return PlayPurchase(purchase_state=0, order_id=order_id, acknowledged=True)


@pytest.mark.asyncio
async def test_valid_purchase_is_entitled_and_persisted(client_and_deps):
    client, fake, session_factory = client_and_deps
    fake.responses["tok-good"] = _purchased()

    resp = await client.post("/api/billing/verify_purchase", json={
        "product_id": PRODUCT, "purchase_token": "tok-good",
    })
    assert resp.status_code == 200
    assert resp.json() == {"entitled": True, "state": "active", "product_id": PRODUCT}

    async with session_factory() as s:
        row = (await s.execute(select(PurchaseEntitlement))).scalar_one()
        assert row.state == "active"
        assert row.purchase_token == "tok-good"
        assert row.order_id == "GPA.1234-5678"


@pytest.mark.asyncio
async def test_fake_token_is_rejected_and_not_persisted(client_and_deps):
    client, fake, session_factory = client_and_deps
    # No response scripted → verifier raises InvalidPurchaseToken

    resp = await client.post("/api/billing/verify_purchase", json={
        "product_id": PRODUCT, "purchase_token": "tok-forged",
    })
    assert resp.status_code == 200
    assert resp.json()["entitled"] is False
    assert resp.json()["state"] == "invalid"

    async with session_factory() as s:
        rows = (await s.execute(select(PurchaseEntitlement))).scalars().all()
        assert rows == []


@pytest.mark.asyncio
async def test_refund_revokes_entitlement_on_recheck(client_and_deps):
    client, fake, session_factory = client_and_deps
    fake.responses["tok-refund"] = _purchased()

    resp = await client.post("/api/billing/verify_purchase", json={
        "product_id": PRODUCT, "purchase_token": "tok-refund",
    })
    assert resp.json()["entitled"] is True

    # Later, Play reports the purchase as cancelled (refunded)
    fake.responses["tok-refund"] = PlayPurchase(
        purchase_state=1, order_id="GPA.1234-5678", acknowledged=True
    )
    resp = await client.get("/api/billing/entitlement", params={
        "product_id": PRODUCT, "purchase_token": "tok-refund",
    })
    assert resp.status_code == 200
    assert resp.json() == {"entitled": False, "state": "revoked", "product_id": PRODUCT}

    async with session_factory() as s:
        row = (await s.execute(select(PurchaseEntitlement))).scalar_one()
        assert row.state == "revoked"
        assert row.revoked_at is not None


@pytest.mark.asyncio
async def test_repurchase_after_refund_reactivates(client_and_deps):
    client, fake, session_factory = client_and_deps
    fake.responses["tok-again"] = PlayPurchase(purchase_state=1, order_id=None, acknowledged=True)
    await client.post("/api/billing/verify_purchase", json={
        "product_id": PRODUCT, "purchase_token": "tok-again",
    })
    # Row only exists if it was previously active; simulate full cycle
    fake.responses["tok-again"] = _purchased()
    await client.post("/api/billing/verify_purchase", json={
        "product_id": PRODUCT, "purchase_token": "tok-again",
    })
    fake.responses["tok-again"] = PlayPurchase(purchase_state=1, order_id=None, acknowledged=True)
    await client.get("/api/billing/entitlement", params={
        "product_id": PRODUCT, "purchase_token": "tok-again",
    })
    fake.responses["tok-again"] = _purchased()
    resp = await client.get("/api/billing/entitlement", params={
        "product_id": PRODUCT, "purchase_token": "tok-again",
    })
    assert resp.json() == {"entitled": True, "state": "active", "product_id": PRODUCT}

    async with session_factory() as s:
        row = (await s.execute(select(PurchaseEntitlement))).scalar_one()
        assert row.state == "active"
        assert row.revoked_at is None


@pytest.mark.asyncio
async def test_pending_purchase_not_entitled(client_and_deps):
    client, fake, _ = client_and_deps
    fake.responses["tok-pending"] = PlayPurchase(purchase_state=2, order_id=None, acknowledged=False)

    resp = await client.post("/api/billing/verify_purchase", json={
        "product_id": PRODUCT, "purchase_token": "tok-pending",
    })
    assert resp.json() == {"entitled": False, "state": "pending", "product_id": PRODUCT}


@pytest.mark.asyncio
async def test_verifier_not_configured_returns_503(client_and_deps):
    client, fake, _ = client_and_deps
    fake.responses["tok-x"] = VerifierNotConfigured("no service account")

    resp = await client.post("/api/billing/verify_purchase", json={
        "product_id": PRODUCT, "purchase_token": "tok-x",
    })
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_play_api_down_returns_502(client_and_deps):
    client, fake, _ = client_and_deps
    fake.responses["tok-y"] = PlayApiError("connection refused")

    resp = await client.post("/api/billing/verify_purchase", json={
        "product_id": PRODUCT, "purchase_token": "tok-y",
    })
    assert resp.status_code == 502
