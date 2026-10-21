#!/usr/bin/env python3
"""CPU-only: reproduce the historical reference sources and patched TK headers."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import zipfile

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
PROVENANCE = json.loads((EXP / "source_provenance.json").read_text())


def verify(folder, expected):
    for name, sha in expected.items():
        path = folder / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise SystemExit(f"Source mismatch: {path}; use a fresh MOK_REPLICA_BUILD_ROOT.")


def materialize(repo, revision, destination, patch, expected):
    if destination.exists():
        verify(destination, expected)
        print(f"Verified existing {destination.name}: {len(expected)} files")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temp:
        temporary = Path(temp)
        raw = subprocess.check_output(["git", "-C", str(repo), "archive", "--format=zip", revision])
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for name in archive.namelist():
                if not (temporary / name).resolve().is_relative_to(temporary):
                    raise SystemExit(f"Unsafe archive member: {name}")
            archive.extractall(temporary)
        # Paths are explicit absolute paths under the generated build directory.
        subprocess.run(["git", "apply", "--unsafe-paths", f"--directory={temporary}", str(patch)],
                       cwd=REPO, check=True)
        verify(temporary, expected)
        temporary.rename(destination)
        # TemporaryDirectory cleanup accepts an already-renamed directory.
    print(f"Prepared {destination.name}: {len(expected)} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("precision", choices=("all", "bf16", "mxfp8"), nargs="?", default="all")
    args = parser.parse_args()
    build = Path(os.environ.get("MOK_REPLICA_BUILD_ROOT", str(EXP / ".build"))).resolve()
    tk = REPO / "third_party/ThunderKittens"
    if not (tk / "include/kittens.cuh").is_file():
        raise SystemExit("Initialize the dependency first: git submodule update --init third_party/ThunderKittens")
    materialize(tk, PROVENANCE["thunderkittens_base_commit"], build / "ThunderKittens",
                EXP / "patches/thunderkittens.patch", PROVENANCE["thunderkittens_files"])
    precisions = ("bf16", "mxfp8") if args.precision == "all" else (args.precision,)
    for precision in precisions:
        materialize(REPO, PROVENANCE["mok_base_commit"], build / f"baseline_{precision}",
                    EXP / f"patches/baseline_{precision}.patch",
                    PROVENANCE["reference_files"][precision])


if __name__ == "__main__":
    main()
