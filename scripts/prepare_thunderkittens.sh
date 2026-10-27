#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tk_root="${repo_root}/third_party/ThunderKittens"
patch_file="${repo_root}/patches/thunderkittens-mxfp8-mn-major.patch"

git -C "${repo_root}" submodule update --init third_party/ThunderKittens

if git -C "${tk_root}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
    echo "ThunderKittens MXFP8 MN-major patch is already applied."
    exit 0
fi

git -C "${tk_root}" apply --check "${patch_file}"
git -C "${tk_root}" apply "${patch_file}"
echo "Applied ThunderKittens MXFP8 MN-major patch."
