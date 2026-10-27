#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
tk_root="${repo_root}/third_party/ThunderKittens"
git -C "${repo_root}" submodule update --init third_party/ThunderKittens

for patch_name in thunderkittens-mxfp8-mn-major thunderkittens-sm107; do
    patch_file="${repo_root}/patches/${patch_name}.patch"
    if git -C "${tk_root}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
        echo "ThunderKittens patch ${patch_name} is already applied."
        continue
    fi
    git -C "${tk_root}" apply --check "${patch_file}"
    git -C "${tk_root}" apply "${patch_file}"
    echo "Applied ThunderKittens patch ${patch_name}."
done
