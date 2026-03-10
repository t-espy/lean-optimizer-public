import copy
import json
import os
import re
from pathlib import Path

from loguru import logger

from optimizer.runner.evaluation import Evaluation

_BASE_CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "base_config.json"

_CLASS_RE = re.compile(
    r"public\s+partial\s+class\s+(\w+)\s*:\s*QCAlgorithm"
)


def _detect_algorithm(artifact_dir: Path, container_path: str) -> dict:
    """Detect algorithm DLL name and type from the compiled artifact directory.

    Finds the DLL, then scans the strategy source (sibling of artifact) for the
    actual QCAlgorithm class name.  Returns overrides dict or empty dict.
    """
    dlls = [p for p in artifact_dir.glob("*.dll") if p.stem != "QuantConnect"]
    if not dlls:
        return {}
    dll = dlls[0]
    dll_name = dll.name
    result = {"algorithm-location": f"{container_path}/{dll_name}"}

    # Try to find the actual class name from the source .cs files
    # The compiler stores strategy_path in a marker file, but we can also
    # check LEAN workspace for a project matching the DLL stem
    workspace = Path(os.environ.get("LEAN_WORKSPACE", "."))
    project_dir = workspace / dll.stem
    if project_dir.is_dir():
        for cs_file in project_dir.glob("*.cs"):
            try:
                text = cs_file.read_text()
            except OSError:
                continue
            m = _CLASS_RE.search(text)
            if m:
                class_name = m.group(1)
                result["algorithm-type-name"] = (
                    f"QuantConnect.Algorithm.CSharp.{class_name}"
                )
                logger.debug(
                    f"[config_builder] Auto-detected algorithm: {dll_name} → {class_name}"
                )
                return result

    logger.warning(
        f"[config_builder] Could not detect class name for {dll_name}; "
        "using base_config default"
    )
    return result


def build(
    evaluation: Evaluation,
    results_root: Path,
    artifacts_container_path: str,
    results_container_path: str,
    artifact_dir: Path | None = None,
) -> Path:
    """Merge base config with per-run overrides; write to results/{eval.id}/config.json."""
    base = json.loads(_BASE_CONFIG_PATH.read_text())
    config = copy.deepcopy(base)

    # Auto-detect algorithm DLL/class from artifact dir if provided
    if artifact_dir is not None:
        overrides = _detect_algorithm(artifact_dir, artifacts_container_path)
        config.update(overrides)

    config["algorithm-id"] = evaluation.id
    config["results-destination-folder"] = f"{results_container_path}/{evaluation.id}"
    config["parameters"] = evaluation.parameters

    run_dir = results_root / evaluation.id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2))
    return config_path
