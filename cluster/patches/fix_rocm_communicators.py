#!/usr/bin/env python3
"""
Patch vLLM's cuda_communicator.py to handle missing CUDA-only dependencies
on ROCm (AMD GPUs).

vLLM's multiproc executor unconditionally imports FlashInfer, CustomAllreduce,
and QuickAllReduce communicators — all CUDA-specific. On ROCm these imports
fail because the underlying CUDA libraries don't exist.

This patch wraps those imports in try/except so vLLM falls back to standard
NCCL/RCCL communication on ROCm.

Usage:
    python3 patches/fix_rocm_communicators.py [--venv /path/to/venv]

If --venv is not specified, uses the currently active virtual environment.
"""

import argparse
import os
import sys


def find_cuda_communicator(venv_path: str) -> str:
    """Find cuda_communicator.py in the given venv."""
    # Try common Python versions
    for pyver in ["python3.10", "python3.11", "python3.12", "python3.13"]:
        path = os.path.join(
            venv_path,
            "lib",
            pyver,
            "site-packages",
            "vllm",
            "distributed",
            "device_communicators",
            "cuda_communicator.py",
        )
        if os.path.exists(path):
            return path
    return ""


OLD_BLOCK = """\
        # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.distributed.device_communicators.flashinfer_all_reduce import (
            FlashInferAllReduce,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )"""

NEW_BLOCK = """\
        # lazy import to avoid documentation build error
        # ROCm patch: wrap CUDA-only communicators in try/except
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        try:
            from vllm.distributed.device_communicators.custom_all_reduce import (
                CustomAllreduce,
            )
        except Exception:
            CustomAllreduce = None
        try:
            from vllm.distributed.device_communicators.flashinfer_all_reduce import (
                FlashInferAllReduce,
            )
        except Exception:
            FlashInferAllReduce = None
        try:
            from vllm.distributed.device_communicators.quick_all_reduce import (
                QuickAllReduce,
            )
        except Exception:
            QuickAllReduce = None"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--venv",
        default=os.environ.get("VIRTUAL_ENV", ""),
        help="Path to the Python venv (default: $VIRTUAL_ENV)",
    )
    args = parser.parse_args()

    if not args.venv:
        print("ERROR: No venv specified and $VIRTUAL_ENV is not set.")
        print("       Activate your venv first or pass --venv /path/to/venv")
        sys.exit(1)

    target = find_cuda_communicator(args.venv)
    if not target:
        print(f"ERROR: cuda_communicator.py not found in {args.venv}")
        sys.exit(1)

    with open(target) as f:
        src = f.read()

    if NEW_BLOCK in src:
        print(f"Already patched: {target}")
        return

    if OLD_BLOCK not in src:
        print(f"ERROR: Expected code block not found in {target}")
        print("       The file may have been modified or vLLM version differs.")
        sys.exit(1)

    with open(target, "w") as f:
        f.write(src.replace(OLD_BLOCK, NEW_BLOCK))

    print(f"Patched: {target}")


if __name__ == "__main__":
    main()
