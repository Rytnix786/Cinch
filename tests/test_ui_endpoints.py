"""Unit tests for Interactive Real-Time Serving Console WebUI (ui/)."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from gateway.app import app


@pytest.mark.asyncio
async def test_ui_static_index_html() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/ui/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Cinch — Real-Time Inference Console" in resp.text
        assert "pane-playground" in resp.text


@pytest.mark.asyncio
async def test_ui_static_css_and_js() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Style
        resp_css = await client.get("/ui/style.css")
        assert resp_css.status_code == 200
        assert "text/css" in resp_css.headers.get("content-type", "") or "text/plain" in resp_css.headers.get(
            "content-type", ""
        )
        assert "--bg-primary" in resp_css.text

        # App JS
        resp_js = await client.get("/ui/app.js")
        assert resp_js.status_code == 200
        assert "Cinch Real-Time Serving Console" in resp_js.text


@pytest.mark.asyncio
async def test_console_redirect() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/console", follow_redirects=False)
        assert resp.status_code in (307, 308, 301, 302)
        assert resp.headers.get("location") == "/ui/"


@pytest.mark.asyncio
async def test_console_state_endpoint() -> None:
    headers = {"Authorization": "Bearer cinch-prod-key"}
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/v1/console/state", headers=headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "uptime_seconds" in data
        assert "queue" in data
        assert "prefix_cache" in data
        assert "semantic_cache" in data
        assert "finops" in data
        assert "shadow_replayer" in data
