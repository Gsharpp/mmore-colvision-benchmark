"""Tests for the generic URL-list PDF downloader. Network is mocked."""

from __future__ import annotations

from pathlib import Path

import httpx

from benchmark_colvision.corpus.url_list_downloader import (
    UrlListItem,
    UrlListManifest,
    download_from_manifest,
    download_one,
)


PDF_MAGIC = b"%PDF-1.4\n%fake content"


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def test_download_one_writes_file(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_MAGIC)

    item = UrlListItem(url="https://e.x/a.pdf", source_id="hal-12345", license="CC-BY")
    with _client(handler) as c:
        out = download_one(item, tmp_path, c, backoff_base_s=0.0)
    assert out.status == "downloaded"
    assert out.pdf_path is not None
    assert out.pdf_path.read_bytes() == PDF_MAGIC
    assert out.sha256 is not None and len(out.sha256) == 64


def test_download_one_idempotent(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, content=PDF_MAGIC)

    item = UrlListItem(url="https://e.x/a.pdf", source_id="hal-1", license="CC-BY")
    with _client(handler) as c:
        first = download_one(item, tmp_path, c, backoff_base_s=0.0)
        second = download_one(item, tmp_path, c, backoff_base_s=0.0)
    assert first.status == "downloaded"
    assert second.status == "cached"
    assert calls["n"] == 1
    assert first.sha256 == second.sha256


def test_download_one_retries_then_succeeds(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, content=b"busy")
        return httpx.Response(200, content=PDF_MAGIC)

    item = UrlListItem(url="https://e.x/a.pdf", source_id="hal-2", license="CC-BY")
    with _client(handler) as c:
        out = download_one(item, tmp_path, c, max_retries=4, backoff_base_s=0.0)
    assert out.status == "downloaded"
    assert calls["n"] == 3


def test_download_one_gives_up(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(500, content=b"boom")

    item = UrlListItem(url="https://e.x/a.pdf", source_id="hal-3", license="CC-BY")
    with _client(handler) as c:
        out = download_one(item, tmp_path, c, max_retries=2, backoff_base_s=0.0)
    assert out.status == "failed"
    assert out.error and "500" in out.error
    assert not (tmp_path / "hal-3.pdf").exists()


def test_target_filename_sanitizes_source_id(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=PDF_MAGIC)

    item = UrlListItem(url="https://e.x/a.pdf", source_id="weird/id with space", license=None)
    with _client(handler) as c:
        out = download_one(item, tmp_path, c, backoff_base_s=0.0)
    assert out.pdf_path is not None
    assert "/" not in out.pdf_path.name
    assert " " not in out.pdf_path.name


def test_download_from_manifest_aggregates_outcomes(tmp_path: Path) -> None:
    counter = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        counter["n"] += 1
        # second item always 404 to exercise the failure path
        if "missing" in str(request.url):
            return httpx.Response(404, content=b"not found")
        return httpx.Response(200, content=PDF_MAGIC)

    manifest = UrlListManifest(
        source="hal",
        language="fr",
        items=[
            UrlListItem(url="https://e.x/a.pdf", source_id="hal-A", license="CC-BY"),
            UrlListItem(url="https://e.x/missing.pdf", source_id="hal-B", license=None),
            UrlListItem(url="https://e.x/c.pdf", source_id="hal-C", license="CC-BY-NC"),
        ],
    )
    with _client(handler) as c:
        report = download_from_manifest(manifest, tmp_path, client=c, max_retries=1)
    assert report.n_total == 3
    assert report.n_downloaded == 2
    assert report.n_failed == 1
    assert report.n_cached == 0
    assert report.source == "hal"
    assert report.language == "fr"


def test_url_list_manifest_roundtrip(tmp_path: Path) -> None:
    m = UrlListManifest(
        source="scielo",
        language="es",
        items=[UrlListItem(url="https://e.x/p.pdf", source_id="scl-1", license="CC-BY")],
    )
    p = tmp_path / "urls.json"
    p.write_text(m.model_dump_json())
    loaded = UrlListManifest.load(p)
    assert loaded.source == "scielo"
    assert loaded.language == "es"
    assert len(loaded.items) == 1
