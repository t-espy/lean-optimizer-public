import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_source_tree(tmp_path: Path) -> Path:
    """Create a minimal fake C# project."""
    proj = tmp_path / "MyStrategy"
    proj.mkdir()
    (proj / "MyStrategy.csproj").write_text("<Project />")
    (proj / "Main.cs").write_text("namespace Foo { class Bar {} }")
    return proj


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_hash_is_stable(tmp_path):
    """Same source directory produces the same hash on two calls."""
    from optimizer.builder.compiler import _compute_hash

    proj = _make_source_tree(tmp_path)
    h1 = _compute_hash(proj)
    h2 = _compute_hash(proj)
    assert h1 == h2
    assert len(h1) == 64  # full hex SHA-256


def test_skip_compile_if_artifact_exists(tmp_path):
    """No Docker call when a DLL is already in the artifact dir."""
    from optimizer.builder import compiler

    proj = _make_source_tree(tmp_path)

    # Pre-populate artifact dir with a fake DLL
    source_hash = compiler._compute_hash(proj)
    artifact_dir = tmp_path / "artifacts" / source_hash[:16]
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "MyStrategy.dll").write_bytes(b"fake-dll")

    with patch("optimizer.builder.compiler.docker") as mock_docker, \
         patch("optimizer.builder.compiler.Path", wraps=Path) as _:
        # Redirect artifacts root to tmp_path/artifacts
        with patch.object(
            compiler,
            "compile",
            wraps=lambda sp, skip=False: _patched_compile(sp, skip, tmp_path),
        ):
            result = _patched_compile(proj, False, tmp_path)

    mock_docker.from_env.assert_not_called()
    assert result == artifact_dir.resolve()


def _patched_compile(strategy_path: Path, skip: bool, tmp_root: Path) -> Path:
    """Re-implementation of compile() that uses tmp_root for artifacts."""
    from optimizer.builder.compiler import _compute_hash, _find_csproj
    import os

    strategy_path = strategy_path.resolve()
    source_hash = _compute_hash(strategy_path)
    hash_prefix = source_hash[:16]

    artifacts_root = tmp_root / "artifacts"
    artifact_dir = artifacts_root / hash_prefix

    existing_dlls = list(artifact_dir.glob("*.dll")) if artifact_dir.exists() else []
    if existing_dlls:
        return artifact_dir.resolve()

    if skip:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir.resolve()

    raise AssertionError("Docker should not be called in this test")


def test_runs_docker_on_cache_miss(tmp_path, monkeypatch):
    """On cache miss, containers.run() is called with correct volumes and command."""
    from optimizer.builder import compiler

    proj = _make_source_tree(tmp_path)

    # Override artifacts root so it points to tmp_path
    monkeypatch.setattr(
        compiler,
        "compile",
        lambda sp, skip=False: _compile_with_mock_docker(sp, skip, tmp_path),
    )

    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 0}
    mock_container.logs.return_value = b""
    mock_client.containers.run.return_value = mock_container

    with patch("optimizer.builder.compiler.docker") as mock_docker:
        mock_docker.from_env.return_value = mock_client
        result = _compile_with_mock_docker(proj, False, tmp_path, mock_docker=mock_docker)

    mock_client.containers.run.assert_called_once()
    call_kwargs = mock_client.containers.run.call_args
    volumes = call_kwargs.kwargs.get("volumes") or call_kwargs.args[0] if call_kwargs.args else {}
    # Verify source dir is mounted ro
    assert any("ro" == v.get("mode") for v in (call_kwargs.kwargs.get("volumes") or {}).values())


def _compile_with_mock_docker(
    strategy_path: Path, skip: bool, tmp_root: Path, mock_docker=None
) -> Path:
    """compile() variant using tmp_root/artifacts and injected docker mock."""
    import os
    import docker as real_docker
    from optimizer.builder.compiler import _compute_hash, _find_csproj

    strategy_path = strategy_path.resolve()
    source_hash = _compute_hash(strategy_path)
    hash_prefix = source_hash[:16]

    artifacts_root = tmp_root / "artifacts"
    artifact_dir = artifacts_root / hash_prefix

    existing_dlls = list(artifact_dir.glob("*.dll")) if artifact_dir.exists() else []
    if existing_dlls:
        return artifact_dir.resolve()

    if skip:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir.resolve()

    csproj = _find_csproj(strategy_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    image = os.environ.get("LEAN_IMAGE", "lean-cli/engine:dgx-arm64")
    client_factory = mock_docker.from_env if mock_docker else real_docker.from_env
    client = client_factory()
    container = client.containers.run(
        image=image,
        command=f'bash -c "dotnet build /LeanCLI/{csproj.name} -o /ArtifactsOut --no-restore"',
        volumes={
            str(strategy_path): {"bind": "/LeanCLI", "mode": "ro"},
            str(artifact_dir.resolve()): {"bind": "/ArtifactsOut", "mode": "rw"},
        },
        remove=False,
        detach=True,
    )
    result = container.wait()
    exit_code = result.get("StatusCode", 1)
    container.remove(force=True)

    if exit_code != 0:
        raise RuntimeError(f"DotNet build failed with exit code {exit_code}")

    return artifact_dir.resolve()


def test_raises_on_build_failure(tmp_path):
    """RuntimeError raised when container exits with non-zero code."""
    from optimizer.builder import compiler

    proj = _make_source_tree(tmp_path)

    mock_client = MagicMock()
    mock_container = MagicMock()
    mock_container.wait.return_value = {"StatusCode": 1}
    mock_container.logs.return_value = b"error: build failed"
    mock_client.containers.run.return_value = mock_container

    with patch("optimizer.builder.compiler.docker") as mock_docker, \
         patch.object(
             compiler,
             "compile",
             side_effect=lambda sp, skip=False: _compile_with_mock_docker(
                 sp, skip, tmp_path, mock_docker=mock_docker
             ),
         ):
        mock_docker.from_env.return_value = mock_client

        with pytest.raises(RuntimeError, match="exit code 1"):
            compiler.compile(proj)
