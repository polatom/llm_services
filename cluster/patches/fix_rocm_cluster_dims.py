#!/usr/bin/env python3
"""
Fix torch._inductor cluster_dims crash on ROCm.

Problem:
    torch/_inductor/runtime/triton_heuristics.py accesses
    `binary.metadata.cluster_dims` unconditionally, but this attribute
    only exists on NVIDIA Triton (Hopper+ CTA clustering). On ROCm Triton
    the attribute is missing, causing:
        'KernelMetadata' object has no attribute 'cluster_dims'

Fix:
    Replace `binary.metadata.cluster_dims` with
    `getattr(binary.metadata, "cluster_dims", (1, 1, 1))`

    The default (1, 1, 1) means "no clustering" which is correct for
    all non-Hopper GPUs including AMD MI210.

Usage:
    python3 cluster/patches/fix_rocm_cluster_dims.py [VENV_PATH]

    VENV_PATH defaults to /lnet/work/people/$USER/.venvs/llm-services-vllm-rocm
"""

import os
import sys
import re

def patch_file(filepath):
    """Patch a single file to use getattr for cluster_dims."""
    with open(filepath, "r") as f:
        content = f.read()

    # Check if already patched
    if 'getattr(binary.metadata, "cluster_dims"' in content:
        print(f"  [SKIP] Already patched: {filepath}")
        return False

    # The buggy line:
    #   (binary.metadata.num_ctas, *binary.metadata.cluster_dims)
    # Replace with:
    #   (binary.metadata.num_ctas, *getattr(binary.metadata, "cluster_dims", (1, 1, 1)))
    old = "binary.metadata.cluster_dims"
    new = 'getattr(binary.metadata, "cluster_dims", (1, 1, 1))'

    if old not in content:
        print(f"  [SKIP] Pattern not found: {filepath}")
        return False

    patched = content.replace(old, new)

    with open(filepath, "w") as f:
        f.write(patched)

    count = content.count(old)
    print(f"  [FIXED] Patched {count} occurrence(s) in: {filepath}")
    return True


def main():
    user = os.environ.get("USER", "tpolak")
    default_venv = f"/lnet/work/people/{user}/.venvs/llm-services-vllm-rocm"

    venv_path = sys.argv[1] if len(sys.argv) > 1 else default_venv

    target = os.path.join(
        venv_path,
        "lib/python3.10/site-packages/torch/_inductor/runtime/triton_heuristics.py"
    )

    print(f"Patching cluster_dims in: {target}")

    if not os.path.exists(target):
        print(f"  [ERROR] File not found: {target}")
        print(f"  Check that the venv path is correct.")
        sys.exit(1)

    if patch_file(target):
        print("\nDone! You can now remove TORCHDYNAMO_DISABLE=1 from run_vllm.sh")
        print("and torch.compile should work on ROCm.")
    else:
        print("\nNo changes needed.")


if __name__ == "__main__":
    main()
