from __future__ import annotations

import shutil
import subprocess
import sys
import os
import re
from pathlib import Path
from typing import Any, Optional


class ProcessService:
    """Run preprocessing and indexing scripts for one uploaded PDF."""
    DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
    ALLOWED_TABLE_CHUNKING_MODES = {"baseline", "row_preserving", "two_stage", "row_blocks"}

    def __init__(self, repo_root: Path, data_root: Path, model_path: Path) -> None:
        self.repo_root = repo_root
        self.data_root = data_root
        self.model_path = model_path

    def _run(self, args: list[str], env: Optional[dict[str, str]] = None) -> str:
        """
        Run a subprocess and return captured stdout/stderr text.

        Raises:
            RuntimeError: If command exits non-zero, including command output.
        """
        proc = subprocess.run(
            args,
            cwd=str(self.repo_root),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        if proc.returncode != 0:
            raise RuntimeError(
                f"Command failed ({proc.returncode}): {' '.join(args)}\n{output}"
            )
        return output

    def process_pdf(self, pdf_path: Path, eval_set_path: Optional[Path] = None) -> dict[str, Any]:
        """
        Execute preprocess + index for an uploaded PDF.

        Returns document metadata used by the API.
        """
        doc_id = pdf_path.stem
        if not self.DOC_ID_RE.fullmatch(str(doc_id)):
            raise ValueError("Invalid derived document id from filename.")
        out_dir = (self.data_root / doc_id).resolve()
        data_root = self.data_root.resolve()
        try:
            out_dir.relative_to(data_root)
        except Exception as e:
            raise ValueError("Unsafe output path for document id.") from e
        out_dir.mkdir(parents=True, exist_ok=True)
        ui_log_path = out_dir / "ui_pipeline.log"
        table_chunking_mode = str(os.getenv("UI_TABLE_CHUNKING_MODE", "baseline") or "baseline").strip()
        if table_chunking_mode not in self.ALLOWED_TABLE_CHUNKING_MODES:
            raise ValueError(
                f"Unsupported UI_TABLE_CHUNKING_MODE={table_chunking_mode!r}. "
                f"Expected one of {sorted(self.ALLOWED_TABLE_CHUNKING_MODES)}."
            )

        cmd_preprocess = [
            sys.executable,
            "preprocess_hybrid.py",
            "--pdf-path",
            str(pdf_path),
            "--out-root",
            str(self.data_root),
            "--table-chunking",
            table_chunking_mode,
        ]
        preprocess_output = self._run(cmd_preprocess)

        if eval_set_path is not None and eval_set_path.exists():
            shutil.copy2(eval_set_path, out_dir / "eval_set.json")

        env = {
            **dict(os.environ),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        cmd_index = [
            sys.executable,
            "scripts/build_index.py",
            "--data-dir",
            str(self.data_root),
            "--model",
            str(self.model_path),
        ]
        index_output = self._run(cmd_index, env=env)
        ui_log_path.write_text(
            "\n".join(
                [
                    "=== PREPROCESS OUTPUT ===",
                    preprocess_output.strip(),
                    "",
                    "=== BUILD INDEX OUTPUT ===",
                    index_output.strip(),
                    "",
                ]
            ),
            encoding="utf-8",
        )

        return {
            "doc_id": doc_id,
            "data_dir": str(out_dir),
            "pipeline_log_path": str(ui_log_path),
            "table_chunking": table_chunking_mode,
        }
