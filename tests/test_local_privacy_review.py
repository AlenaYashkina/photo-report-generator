from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image

from examples.privacy_review_contact_sheet import create_review


def test_local_privacy_contact_sheets():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory) / "example"
        root.mkdir()
        Image.new("RGB", (100, 80), "blue").save(root / "fictional.jpg")
        Image.new("RGB", (100, 80), "red").save(root / "fictional_filtered.jpg")
        (root / "fictional.pptx").write_bytes(b"synthetic placeholder, reviewed separately")
        output = Path(directory) / ".privacy-review"
        photos, documents = create_review(root, output)
        assert (photos, documents) == (1, 1)
        assert (output / "contact-sheet-01.jpg").exists()
        manifest = (output / "manual-review-checklist.csv").read_text(encoding="utf-8-sig")
        assert "fictional.jpg" in manifest
        assert "fictional.pptx" in manifest
        assert "fictional_filtered.jpg" not in manifest


if __name__ == "__main__":
    test_local_privacy_contact_sheets()
    print("PASS local privacy contact-sheet generator")
