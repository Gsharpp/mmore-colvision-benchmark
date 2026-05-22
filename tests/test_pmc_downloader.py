"""Tests for the PMC OA index parser and tarball extraction.

Network calls are stubbed.
"""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from benchmark_colvision.corpus.pmc_downloader import (
    PmcPackage,
    extract_pdfs,
    iter_package_index,
    parse_mesh_terms_from_nxml,
)


SAMPLE_CSV = """File,Article Citation,Accession ID,Last Updated,PMID,License
oa_package/00/01/PMC1234567.tar.gz,J Foo. 2020 1(1):1-2.,PMC1234567,2020-01-01,1111,CC-BY
oa_package/00/02/PMC2345678.tar.gz,J Bar. 2020 1(1):3-4.,PMC2345678,2020-01-01,2222,CC-BY-NC
oa_package/00/03/PMC3456789.tar.gz,J Baz. 2020 1(1):5-6.,PMC3456789,2020-01-01,3333,NO-CC
"""


def test_iter_package_index_no_filter() -> None:
    pkgs = list(iter_package_index(SAMPLE_CSV))
    assert len(pkgs) == 3
    assert pkgs[0].pmcid == "PMC1234567"
    assert pkgs[0].relative_path == "oa_package/00/01/PMC1234567.tar.gz"
    assert pkgs[0].license == "CC-BY"


def test_iter_package_index_filtered() -> None:
    pkgs = list(iter_package_index(SAMPLE_CSV, pmcid_filter={"PMC2345678"}))
    assert len(pkgs) == 1
    assert pkgs[0].pmcid == "PMC2345678"


def test_iter_package_index_url_property() -> None:
    pkg = PmcPackage(pmcid="PMC1", relative_path="oa_package/00/01/PMC1.tar.gz", license="CC-BY")
    assert pkg.url.endswith("oa_package/00/01/PMC1.tar.gz")


def test_iter_package_index_bad_header_raises() -> None:
    bad = "Wrong,Header\nrow1,row2\n"
    with pytest.raises(ValueError):
        list(iter_package_index(bad))


def test_extract_pdfs(tmp_path: Path) -> None:
    archive = tmp_path / "package.tar.gz"
    pdf_bytes = b"%PDF-1.4 fake"
    nxml_bytes = b"<article/>"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="PMC1234567/main.pdf")
        info.size = len(pdf_bytes)
        tf.addfile(info, io.BytesIO(pdf_bytes))
        info = tarfile.TarInfo(name="PMC1234567/main.nxml")
        info.size = len(nxml_bytes)
        tf.addfile(info, io.BytesIO(nxml_bytes))
    archive.write_bytes(buf.getvalue())

    out_dir = tmp_path / "pdfs"
    pdfs = extract_pdfs(archive, out_dir)
    assert len(pdfs) == 1
    assert pdfs[0].name == "main.pdf"
    assert pdfs[0].read_bytes() == pdf_bytes


def test_parse_mesh_terms_from_nxml() -> None:
    nxml = """<article>
    <front><kwd-group><kwd>Neoplasms</kwd><kwd>MRI</kwd></kwd-group></front>
    </article>"""
    tags = parse_mesh_terms_from_nxml(nxml)
    assert tags == ["Neoplasms", "MRI"]


def test_parse_mesh_terms_invalid_xml() -> None:
    assert parse_mesh_terms_from_nxml("<not really xml") == []
