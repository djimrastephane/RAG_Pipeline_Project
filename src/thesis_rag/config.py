from __future__ import annotations

from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import yaml

from .schemas import (
    BM25Config,
    ChunkingConfig,
    EmbeddingConfig,
    EvaluationConfig,
    FaissConfig,
    OCRConfig,
    PathsConfig,
    PipelineConfig,
    RetrievalConfig,
    RuntimeConfig,
)

T = TypeVar("T")


def _coerce_value(field_type: Any, value: Any, *, base_dir: Path) -> Any:
    origin = get_origin(field_type)
    if field_type is Path:
        return Path(value)
    if origin is list:
        inner = get_args(field_type)[0]
        return [_coerce_value(inner, item, base_dir=base_dir) for item in value]
    if is_dataclass(field_type):
        return _build_dataclass(field_type, value or {}, base_dir=base_dir)
    return value


def _build_dataclass(cls: type[T], payload: dict[str, Any], *, base_dir: Path) -> T:
    template = cls()
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in payload:
            continue
        current_value = getattr(template, item.name)
        raw_value = payload[item.name]
        if isinstance(current_value, Path):
            kwargs[item.name] = Path(raw_value)
            continue
        if is_dataclass(current_value):
            kwargs[item.name] = _build_dataclass(type(current_value), raw_value or {}, base_dir=base_dir)
            continue
        kwargs[item.name] = _coerce_value(item.type, raw_value, base_dir=base_dir)
    return cls(**kwargs)


def _resolve_paths(config: PipelineConfig, *, config_path: Path) -> PipelineConfig:
    base_dir = config_path.resolve().parent
    for name in (
        "project_root",
        "data_dir",
        "processed_dir",
        "indexes_dir",
        "runs_dir",
        "query_set_path",
        "model_cache_dir",
    ):
        value = getattr(config.paths, name)
        path = Path(value)
        if not path.is_absolute():
            setattr(config.paths, name, (base_dir / path).resolve())
    model_name = config.embedding.model_name
    model_path = Path(model_name)
    if any(sep in model_name for sep in ("/", "\\")) and not model_path.is_absolute():
        config.embedding.model_name = str((base_dir / model_path).resolve())
    return config


def load_config(config_path: str | Path) -> PipelineConfig:
    path = Path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = PipelineConfig(
        paths=_build_dataclass(PathsConfig, payload.get("paths", {}), base_dir=path.parent),
        runtime=_build_dataclass(RuntimeConfig, payload.get("runtime", {}), base_dir=path.parent),
        ocr=_build_dataclass(OCRConfig, payload.get("ocr", {}), base_dir=path.parent),
        chunking=_build_dataclass(ChunkingConfig, payload.get("chunking", {}), base_dir=path.parent),
        embedding=_build_dataclass(EmbeddingConfig, payload.get("embedding", {}), base_dir=path.parent),
        faiss=_build_dataclass(FaissConfig, payload.get("faiss", {}), base_dir=path.parent),
        bm25=_build_dataclass(BM25Config, payload.get("bm25", {}), base_dir=path.parent),
        retrieval=_build_dataclass(RetrievalConfig, payload.get("retrieval", {}), base_dir=path.parent),
        evaluation=_build_dataclass(EvaluationConfig, payload.get("evaluation", {}), base_dir=path.parent),
    )
    return _resolve_paths(config, config_path=path)
