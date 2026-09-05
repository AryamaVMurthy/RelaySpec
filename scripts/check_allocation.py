from __future__ import annotations

import json
import os
import platform
import sys
import time
from importlib.metadata import PackageNotFoundError, version

import torch

EXPECTED_GPUS = 4


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    visible_count = torch.cuda.device_count()
    if visible_count != EXPECTED_GPUS:
        raise RuntimeError(
            f"expected exactly 4 visible CUDA devices, found {visible_count}; "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the Slurm allocation")

    devices: list[dict[str, object]] = []
    for index in range(visible_count):
        properties = torch.cuda.get_device_properties(index)
        start = time.perf_counter()
        left = torch.randn((2048, 2048), device=f"cuda:{index}", dtype=torch.float16)
        right = torch.randn((2048, 2048), device=f"cuda:{index}", dtype=torch.float16)
        product = left @ right
        torch.cuda.synchronize(index)
        elapsed_ms = (time.perf_counter() - start) * 1_000
        if not torch.isfinite(product).all().item():
            raise RuntimeError(f"non-finite matrix result on cuda:{index}")
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "matmul_elapsed_ms": elapsed_ms,
                "matmul_checksum": float(product[0, 0].item()),
            }
        )

    peer_access = [
        [
            index == peer or torch.cuda.can_device_access_peer(index, peer)
            for peer in range(visible_count)
        ]
        for index in range(visible_count)
    ]
    result = {
        "status": "pass",
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "transformers_version": package_version("transformers"),
        "tokenizers_version": package_version("tokenizers"),
        "numpy_version": package_version("numpy"),
        "device_count": visible_count,
        "devices": devices,
        "peer_access": peer_access,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
