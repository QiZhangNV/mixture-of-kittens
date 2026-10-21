#!/usr/bin/env python3
"""CPU-only validation of the checked-in experiment and tested device sources."""
import hashlib
import json
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
manifest = json.loads((EXP / "MANIFEST.sha256.json").read_text())
provenance = json.loads((EXP / "source_provenance.json").read_text())
for name, digest in manifest.items():
    path = (REPO / name).resolve()
    assert path.is_relative_to(REPO) and path.is_file(), name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
for name, digest in provenance["fused_files"].items():
    if name == "README.md":  # Root documentation adds a discoverability link.
        continue
    assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest, name
print(f"Verified {len(manifest)} checked-in files and the historical fused source hashes")
