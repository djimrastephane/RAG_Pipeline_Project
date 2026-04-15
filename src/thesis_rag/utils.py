from __future__ import annotations

import importlib
import json
import logging
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def configure_logging(log_path: Path, level: str = "INFO") -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def set_global_determinism(seed: int, deterministic_torch: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def resolve_device(requested: str) -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    desired = requested.strip().lower()
    if desired == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if desired == "cuda" and not torch.cuda.is_available():
        return "cpu"
    if desired == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        return "cpu"
    return desired or "cpu"


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, eps)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit_hash(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            check=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def make_run_dir(runs_dir: Path, prefix: str = "run") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / f"{timestamp}_{prefix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dependency_check(required: dict[str, str]) -> None:
    missing: list[str] = []
    mismatched: list[str] = []
    for module_name, expected_version in required.items():
        try:
            module = importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
            continue
        actual = getattr(module, "__version__", None)
        if expected_version and actual and actual != expected_version:
            mismatched.append(f"{module_name}=={actual} (expected {expected_version})")
    if missing or mismatched:
        parts: list[str] = []
        if missing:
            parts.append(f"missing: {', '.join(sorted(missing))}")
        if mismatched:
            parts.append(f"version mismatch: {', '.join(sorted(mismatched))}")
        raise RuntimeError("Dependency validation failed: " + "; ".join(parts))


def stable_unique(values: list[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered
