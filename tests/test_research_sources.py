import hashlib

from src.research_sources import sha256_file, source_kind


def test_sha256_file_reads_content_without_modifying_source(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"ultrasound-research-source")
    before = source.stat().st_mtime_ns

    digest = sha256_file(source)

    assert digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert source.stat().st_mtime_ns == before


def test_generated_mask_png_is_not_counted_as_an_input_image(tmp_path):
    mask = tmp_path / "patient" / "mask" / "example.png"

    assert source_kind(mask) == "annotation_mask"
    assert source_kind(tmp_path / "patient" / "example.jpg") == "image"
