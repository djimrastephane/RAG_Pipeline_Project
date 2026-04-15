from pathlib import Path

from thesis_rag.config import load_config


def test_load_config_resolves_relative_embedding_model_path(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    configs_dir = project_root / "configs"
    models_dir = project_root / "models" / "all-MiniLM-L6-v2"
    configs_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    config_path = configs_dir / "test.yaml"
    config_path.write_text(
        "\n".join(
            [
                "paths:",
                "  project_root: ..",
                "  data_dir: ../data",
                "  processed_dir: ../processed",
                "  indexes_dir: ../indexes",
                "  runs_dir: ../runs",
                "  query_set_path: ../data/eval_set.json",
                "  model_cache_dir: ../models",
                "embedding:",
                "  model_name: ../models/all-MiniLM-L6-v2",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.embedding.model_name == str(models_dir.resolve())
