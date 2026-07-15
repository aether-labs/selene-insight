"""Regression tests for the rate-limited Space-Track fetch path."""

import asyncio


def test_fetch_once_uses_space_track_contains_filter(monkeypatch):
    from services.telemetry import spacetrack_fetcher

    requests = []

    class FakeResponse:
        text = "0 STARLINK-1008\nline1\nline2\n"

        def raise_for_status(self):
            pass

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url, **kwargs):
            requests.append(("POST", url))
            return FakeResponse()

        async def get(self, url, **kwargs):
            requests.append(("GET", url))
            return FakeResponse()

    monkeypatch.setattr(spacetrack_fetcher.httpx, "AsyncClient", FakeClient)

    text = asyncio.run(spacetrack_fetcher._fetch_once("user", "password"))

    assert "/OBJECT_NAME/STARLINK~~/" in requests[1][1]
    assert "0 STARLINK" not in text
    assert "STARLINK-1008" in text
