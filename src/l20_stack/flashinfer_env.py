"""Runtime environment helpers for FlashInfer JIT kernels on L20."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import ctypes
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class FlashInferCudaEnv:
    cuda_home: str
    nvcc: str
    nvcc_version: str
    library_paths: tuple[str, ...]
    preloaded_libraries: tuple[str, ...]
    changed: bool

    def to_dict(self):
        return asdict(self)


def _python_site_roots() -> Iterable[Path]:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    yield Path(sys.prefix) / "lib" / version / "site-packages"
    yield Path(sys.prefix) / "Lib" / "site-packages"
    for entry in sys.path:
        if entry:
            yield Path(entry)


def _candidate_cuda_roots() -> Iterable[Path]:
    env_home = os.environ.get("L20_FLASHINFER_CUDA_HOME") or os.environ.get("CUDA_HOME")
    if env_home:
        yield Path(env_home)
    for root in _python_site_roots():
        yield root / "nvidia" / "cu13"
    yield Path("/usr/local/cuda-13.0")
    yield Path("/usr/local/cuda")


def _nvcc_version(nvcc: Path) -> str:
    try:
        result = subprocess.run(
            [str(nvcc), "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return result.stdout


def _is_cuda13(version_text: str) -> bool:
    return bool(re.search(r"release\s+13\.", version_text))


def find_cuda13_root() -> Optional[Path]:
    """Find a CUDA 13 toolkit root usable by FlashInfer's JIT build."""

    seen = set()
    for root in _candidate_cuda_roots():
        root = root.expanduser().resolve()
        if root in seen:
            continue
        seen.add(root)
        nvcc = root / "bin" / "nvcc"
        if nvcc.exists() and _is_cuda13(_nvcc_version(nvcc)):
            return root
    return None


def _preload_first_library(library_paths: Iterable[str], names: Iterable[str]) -> Optional[str]:
    for directory in library_paths:
        for name in names:
            candidate = Path(directory) / name
            if candidate.exists():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                return str(candidate)
    return None


def _preload_cuda_runtime_libraries(library_paths: Iterable[str]) -> tuple[str, ...]:
    """Load CUDA runtime libraries before FlashInfer JIT modules call dlopen.

    Updating LD_LIBRARY_PATH inside a running Python process is too late for
    some dynamic-loader paths. Preloading by absolute path makes the CUDA 13
    wheel runtime visible to FlashInfer's generated shared objects.
    """

    loaded = []
    for names in (
        ("libcudart.so.13", "libcudart.so"),
        ("libcurand.so.10", "libcurand.so"),
    ):
        path = _preload_first_library(library_paths, names)
        if path is not None:
            loaded.append(path)
    return tuple(loaded)


def configure_flashinfer_cuda13_env(required: bool = True) -> Optional[FlashInferCudaEnv]:
    """Ensure FlashInfer JIT subprocesses see CUDA 13 nvcc first.

    The L20 remote host can have a CUDA 12 system nvcc while the Python stack is
    built against CUDA 13. FlashInfer 0.6.x sampling JIT then fails while
    compiling its vendored CCCL/CUB sources. This helper points CUDA_HOME,
    CUDACXX, PATH, and CUDA library search paths at a CUDA 13 toolkit before
    importing FlashInfer sampling.
    """

    root = find_cuda13_root()
    if root is None:
        if required:
            raise RuntimeError(
                "CUDA 13 nvcc is required for FlashInfer sampling JIT. "
                "Install nvidia-cuda-nvcc in the active environment or set "
                "L20_FLASHINFER_CUDA_HOME to a CUDA 13 toolkit root."
            )
        return None
    nvcc = root / "bin" / "nvcc"
    version = _nvcc_version(nvcc)
    old_path = os.environ.get("PATH", "")
    bin_path = str(root / "bin")
    python_bin_path = str(Path(sys.executable).parent)
    path_entries = [entry for entry in old_path.split(os.pathsep) if entry]
    cuda_library_paths = tuple(
        str(path)
        for path in (
            root / "lib64",
            root / "lib",
            root / "targets" / "x86_64-linux" / "lib",
            root / "lib64" / "stubs",
        )
        if path.exists()
    )
    changed = False
    if os.environ.get("CUDA_HOME") != str(root):
        os.environ["CUDA_HOME"] = str(root)
        changed = True
    if os.environ.get("CUDACXX") != str(nvcc):
        os.environ["CUDACXX"] = str(nvcc)
        changed = True
    desired_path = []
    for entry in (bin_path, python_bin_path, *path_entries):
        if entry not in desired_path:
            desired_path.append(entry)
    new_path = os.pathsep.join(desired_path)
    if os.environ.get("PATH", "") != new_path:
        os.environ["PATH"] = new_path
        changed = True
    for env_name in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        old_entries = [entry for entry in os.environ.get(env_name, "").split(os.pathsep) if entry]
        desired_entries = []
        for entry in (*cuda_library_paths, *old_entries):
            if entry not in desired_entries:
                desired_entries.append(entry)
        new_entries = os.pathsep.join(desired_entries)
        if os.environ.get(env_name, "") != new_entries:
            os.environ[env_name] = new_entries
            changed = True
    preloaded_libraries = _preload_cuda_runtime_libraries(cuda_library_paths)
    return FlashInferCudaEnv(
        cuda_home=str(root),
        nvcc=str(nvcc),
        nvcc_version=version.strip(),
        library_paths=cuda_library_paths,
        preloaded_libraries=preloaded_libraries,
        changed=changed,
    )
