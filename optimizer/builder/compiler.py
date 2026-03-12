import hashlib
import os
import re
import subprocess
import tempfile
from pathlib import Path

import docker
from docker.types import Mount
from loguru import logger

LEAN_ROOT_PATH = "/Lean/Launcher/bin/Debug"

_DIRECTORY_BUILD_PROPS = """\
<Project>
    <PropertyGroup>
        <BaseIntermediateOutputPath>/Compile/obj/$(MSBuildProjectName)/</BaseIntermediateOutputPath>
        <IntermediateOutputPath>/Compile/obj/$(MSBuildProjectName)/</IntermediateOutputPath>
        <DefaultItemExcludes>$(DefaultItemExcludes);backtests/*/code/**;live/*/code/**;optimizations/*/code/**</DefaultItemExcludes>
        <NoWarn>CS0618</NoWarn>
    </PropertyGroup>
</Project>"""


def _compute_hash(strategy_path: Path) -> str:
    """SHA-256 of sorted concatenated content of top-level *.cs and *.csproj files.

    Mirrors the .csproj DefaultItemExcludes: backtests/*, optimizations/*, live/*
    are snapshot dirs written by the lean CLI and must not affect the build hash.
    """
    excluded = {"backtests", "optimizations", "live"}

    def _not_excluded(p: Path) -> bool:
        parts = p.relative_to(strategy_path).parts
        return not (len(parts) > 1 and parts[0] in excluded)

    source_files = sorted(
        f for f in
        list(strategy_path.glob("**/*.cs")) + list(strategy_path.glob("**/*.csproj"))
        if _not_excluded(f)
    )
    hasher = hashlib.sha256()
    for f in source_files:
        hasher.update(f.read_bytes())
    return hasher.hexdigest()


def _find_csproj(strategy_path: Path) -> Path:
    matches = list(strategy_path.glob("*.csproj"))
    if not matches:
        raise FileNotFoundError(f"No .csproj found in {strategy_path}")
    return matches[0]


def _patch_csproj(csproj_path: Path) -> str:
    """Return csproj XML with QuantConnect NuGet ref replaced by local LEAN DLL reference."""
    content = csproj_path.read_text(encoding="utf-8")

    # Replace <PackageReference Include="QuantConnect.*" ...> (self-closing or with children)
    # with a <Reference> pointing to the container's LEAN DLLs.
    patched, count = re.subn(
        r'<PackageReference\s+Include="QuantConnect\.[^"]*"[^/]*/?>',
        f'<Reference Include="{LEAN_ROOT_PATH}/*.dll"><Private>False</Private></Reference>',
        content,
        flags=re.IGNORECASE,
    )
    if count == 0:
        logger.warning("[compiler] No QuantConnect PackageReference found in csproj — building as-is")
    else:
        logger.debug(f"[compiler] Patched {count} QuantConnect PackageReference(s) → local DLLs")
    return patched


def compile(strategy_path: Path, skip: bool = False) -> Path:
    """Compile C# strategy once; return absolute path to artifact dir with DLL.

    The lean-cli engine image has ENTRYPOINT set to the LEAN Launcher.  We
    override it with ``dotnet build`` and patch the .csproj to reference the
    LEAN DLLs already inside the container instead of the NuGet package
    (which may target a newer .NET framework than the image SDK).

    Args:
        strategy_path: Path to the C# project directory.
        skip: If True, skip compile even on cache miss (e.g. for dry-runs).

    Returns:
        Absolute path to artifact directory containing the compiled DLL.
    """
    strategy_path = strategy_path.resolve()
    source_hash = _compute_hash(strategy_path)
    hash_prefix = source_hash[:16]

    artifacts_root = Path(__file__).parent.parent.parent / "artifacts"
    artifact_dir = artifacts_root / hash_prefix

    existing_dlls = list(artifact_dir.glob("*.dll")) if artifact_dir.exists() else []
    if existing_dlls:
        logger.info(f"[compiler] Cache hit {hash_prefix} — skipping build")
        return artifact_dir.resolve()

    if skip:
        logger.info("[compiler] --skip-compile set; returning artifact dir without building")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir.resolve()

    csproj = _find_csproj(strategy_path)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    image = os.environ.get("LEAN_IMAGE", "lean-cli/engine:dgx-arm64")
    logger.info(f"[compiler] Building {csproj.name} → artifacts/{hash_prefix}/")

    with tempfile.TemporaryDirectory(prefix="lean_compile_") as tmp:
        tmp_path = Path(tmp)

        # Patched csproj — replaces NuGet ref with in-container DLL glob
        patched_csproj = tmp_path / csproj.name
        patched_csproj.write_text(_patch_csproj(csproj), encoding="utf-8")

        # Directory.Build.props — redirects obj/ to /Compile/obj (writable in container)
        build_props = tmp_path / "Directory.Build.props"
        build_props.write_text(_DIRECTORY_BUILD_PROPS, encoding="utf-8")

        msbuild = (
            "Configuration=Release"
            ";Platform=AnyCPU"
            ";TargetFramework=net9.0"
            f";OutputPath=/ArtifactsOut"
            ";GenerateAssemblyInfo=false"
            ";AppendTargetFrameworkToOutputPath=false"
            ";CopyLocalLockFileAssemblies=true"
        )

        client = docker.from_env()
        container = client.containers.run(
            image=image,
            entrypoint=["dotnet", "build"],
            command=[f"/LeanCLI/{csproj.name}", f"-p:{msbuild}"],
            environment={
                "DOTNET_NOLOGO": "true",
                "DOTNET_CLI_TELEMETRY_OPTOUT": "true",
            },
            volumes={
                str(strategy_path): {"bind": "/LeanCLI", "mode": "rw"},
                str(artifact_dir.resolve()): {"bind": "/ArtifactsOut", "mode": "rw"},
            },
            mounts=[
                # Overlay patched csproj on top of the rw-mounted source dir
                Mount(
                    target=f"/LeanCLI/{csproj.name}",
                    source=str(patched_csproj),
                    type="bind",
                    read_only=True,
                ),
                # Directory.Build.props at filesystem root — MSBuild walks up to find it
                Mount(
                    target="/Directory.Build.props",
                    source=str(build_props),
                    type="bind",
                    read_only=True,
                ),
            ],
            remove=False,
            detach=True,
        )

        # Stream build output in real-time; generator exhausts on container exit
        for chunk in container.logs(stream=True, follow=True):
            line = chunk.decode("utf-8", errors="replace").rstrip()
            if line:
                logger.debug(f"[build] {line}")

        result = container.wait()
        exit_code = result.get("StatusCode", 1)
        container.remove(force=True)

    if exit_code != 0:
        raise RuntimeError(f"DotNet build failed with exit code {exit_code}")

    built_dlls = list(artifact_dir.glob("*.dll"))
    if not built_dlls:
        raise RuntimeError(f"Build reported success but no DLL found in {artifact_dir}")

    # Fix ownership — Docker runs as root, so artifacts are root-owned
    uid, gid = os.getuid(), os.getgid()
    for f in artifact_dir.iterdir():
        try:
            os.chown(f, uid, gid)
        except OSError:
            pass

    logger.success(f"[compiler] Build succeeded → artifacts/{hash_prefix}/ ({built_dlls[0].name})")
    return artifact_dir.resolve()


_HARNESS_CSPROJ = Path(__file__).parent.parent.parent / "harness" / "LeanHarness" / "LeanHarness.csproj"
_HARNESS_OUTPUT = _HARNESS_CSPROJ.parent / "bin" / "Release"


def compile_harness(force: bool = False) -> Path:
    """Build the persistent LEAN worker harness on the host.

    Returns:
        Absolute path to the harness output directory (bin/Debug/).
    """
    harness_dll = _HARNESS_OUTPUT / "LeanHarness.dll"

    if harness_dll.exists() and not force:
        logger.info("[compiler] Harness already built — skipping")
        return _HARNESS_OUTPUT.resolve()

    if not _HARNESS_CSPROJ.exists():
        raise FileNotFoundError(f"Harness csproj not found: {_HARNESS_CSPROJ}")

    logger.info("[compiler] Building harness...")
    result = subprocess.run(
        ["dotnet", "build", str(_HARNESS_CSPROJ), "-c", "Release"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"[compiler] Harness build failed:\n{result.stderr}")
        raise RuntimeError(f"Harness build failed (exit {result.returncode})")

    if not harness_dll.exists():
        raise RuntimeError(f"Harness build succeeded but DLL not found: {harness_dll}")

    logger.success("[compiler] Harness build succeeded")
    return _HARNESS_OUTPUT.resolve()
