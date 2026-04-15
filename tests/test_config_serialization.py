from pathlib import Path

from thesis_rag.schemas import PipelineConfig


def test_pipeline_config_to_dict_converts_paths_to_strings() -> None:
    config = PipelineConfig()
    config.paths.project_root = Path("/tmp/project")
    config.paths.data_dir = Path("/tmp/project/data")
    config.paths.query_set_path = Path("/tmp/project/data/eval_set.json")

    payload = config.to_dict()

    assert payload["paths"]["project_root"] == "/tmp/project"
    assert payload["paths"]["data_dir"] == "/tmp/project/data"
    assert payload["paths"]["query_set_path"] == "/tmp/project/data/eval_set.json"
    assert isinstance(payload["paths"]["runs_dir"], str)
