#!/usr/bin/env python3
"""lean-optimizer — Parallel Batch Runner & Optimization Pipeline

Usage:
    python main.py batch              [--strategy-path PATH] [--skip-compile]
                                      [--parameter-sets PATH] [--worker-count N]
    python main.py backtest           [--strategy-path PATH] [--skip-compile]
                                      (--param KEY=VALUE ... | --params-file PATH)
                                      [--worker-count N]
    python main.py optimize           [--strategy-path PATH] [--skip-compile]
                                      [--config PATH] [--worker-count N]
    python main.py optimize-universe  [--strategy-path PATH] [--skip-compile]
                                      --universe PATH
    python main.py backtest-universe  [--strategy-path PATH] [--skip-compile]
                                      --universe PATH
"""
import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Default stderr handler: INFO level (loguru starts with DEBUG)
logger.remove()
logger.add(sys.stderr, level="INFO")

from optimizer.builder import compiler
from optimizer.docker.pool import WorkerPool
from optimizer.runner.backtest_runner import run_backtests
from optimizer.runner.batch_runner import run_batch

PROJECT_ROOT = Path(__file__).parent
DEFAULT_STRATEGY = Path("your_strategy")
DEFAULT_PARAM_SETS = PROJECT_ROOT / "config" / "parameter_sets.json"
DEFAULT_OPT_CONFIG = PROJECT_ROOT / "config" / "optimization_ga.json"
ARTIFACTS_CONTAINER_PATH = os.environ.get("ARTIFACTS_CONTAINER_PATH", "/Artifacts")
BACKTESTS_ROOT = PROJECT_ROOT / "backtests"


def _resolve_results_root() -> Path:
    """Resolve results root from RESULTS_SCRATCH_PATH; fall back to ./results."""
    scratch = os.environ.get("RESULTS_SCRATCH_PATH", "").strip()
    if scratch:
        path = Path(scratch)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok")
            probe.unlink()
            logger.info(f"Results scratch: {path}")
            return path.resolve()
        except Exception as e:
            logger.warning(
                f"RESULTS_SCRATCH_PATH {scratch!r} not accessible ({e}) — "
                f"falling back to ./results"
            )
    fallback = (PROJECT_ROOT / "results").resolve()
    fallback.mkdir(exist_ok=True)
    logger.info(f"Results root: {fallback} (fallback)")
    return fallback


# ---------------------------------------------------------------------------
# Batch mode output
# ---------------------------------------------------------------------------


def _print_batch_summary(evaluations: list, wall_seconds: float) -> None:
    header = (
        f"{'UUID':8}  {'Worker':20}  {'ComprLB':>7}  {'Thresh':>7}"
        f"  {'Status':8}  {'Runtime':>9}  {'Sharpe':>8}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(header)
    print("-" * len(header))
    for ev in sorted(evaluations, key=lambda e: e.id):
        uid = ev.id[:8]
        wid = ev.worker_id or "n/a"
        worker = wid.split("_")[-1] if "_" in wid else wid[-8:]
        lb = ev.parameters.get("CompressionLookback", "?")
        thresh = ev.parameters.get("CompressionThreshold", "?")
        status = ev.status.value
        runtime = f"{ev.runtime_seconds:.1f}s" if ev.runtime_seconds is not None else "n/a"
        sharpe = "n/a"
        if ev.metrics:
            sharpe = ev.metrics.get("Sharpe Ratio", ev.metrics.get("SharpeRatio", "n/a"))
        print(
            f"{uid:8}  {worker:20}  {lb:>7}  {thresh:>7}"
            f"  {status:8}  {runtime:>9}  {sharpe!s:>8}"
        )
    print("-" * len(header))
    success = sum(1 for e in evaluations if e.status.value == "success")
    print(f"Wall: {wall_seconds:.1f}s  |  {success}/{len(evaluations)} succeeded")
    print(f"{sep}\n")


def _print_backtest_summary(evaluations: list, wall_seconds: float) -> None:
    ok = sum(1 for e in evaluations if e.status.value == "success")
    print(f"\nBacktests complete: {ok}/{len(evaluations)} succeeded")
    print(f"Wall time: {wall_seconds:.1f}s\n")


# ---------------------------------------------------------------------------
# Optimize mode output
# ---------------------------------------------------------------------------


def _print_optimize_summary(
    pipeline_result, wall_seconds: float, min_trades: int = 100,
) -> None:

    print(f"\n{'='*70}")
    print("  OPTIMIZATION RESULTS")
    print(f"{'='*70}")

    # Section 1: LHS summary
    lhs_evals = pipeline_result.stage_results.get("LHSStage", [])
    if lhs_evals:
        valid = [e for e in lhs_evals if e.fitness_score is not None and e.fitness_score != float("-inf")]
        best_lhs = max(valid, key=lambda e: e.fitness_score) if valid else None
        print(f"\n  LHS Stage: {len(lhs_evals)} sampled, {len(valid)} valid")
        if best_lhs:
            print(f"    Best: {best_lhs.fitness_score:.4f}  params={best_lhs.parameters}")

    # Section 2: Bayesian summary with delta
    bay_evals = pipeline_result.stage_results.get("BayesianStage", [])
    if bay_evals:
        valid = [e for e in bay_evals if e.fitness_score is not None and e.fitness_score != float("-inf")]
        best_bay = max(valid, key=lambda e: e.fitness_score) if valid else None
        print(f"\n  Bayesian Stage: {len(bay_evals)} evaluated, {len(valid)} valid")
        if best_bay:
            delta = ""
            if best_lhs and best_lhs.fitness_score != float("-inf"):
                d = best_bay.fitness_score - best_lhs.fitness_score
                delta = f"  (delta from LHS: {d:+.4f})"
            print(f"    Best: {best_bay.fitness_score:.4f}{delta}  params={best_bay.parameters}")

    # Section 3: GA summary (convergence)
    ga_evals = pipeline_result.stage_results.get("GeneticStage", [])
    if ga_evals:
        valid = [e for e in ga_evals if e.fitness_score is not None and e.fitness_score != float("-inf")]
        best_ga = max(valid, key=lambda e: e.fitness_score) if valid else None
        print(f"\n  Genetic Algorithm: {len(ga_evals)} evaluated, {len(valid)} valid")
        if best_ga:
            print(f"    Best: {best_ga.fitness_score:.4f}  params={best_ga.parameters}")
        # Convergence: cumulative best fitness across generations
        gens = {}
        for ev in ga_evals:
            g = ev.stage_detail or "?"
            if ev.fitness_score is not None and ev.fitness_score != float("-inf"):
                gens.setdefault(g, []).append(ev.fitness_score)
        if gens:
            print("    Convergence:")
            cumulative_best = float("-inf")
            for g in sorted(gens.keys(), key=lambda k: (-1, 0) if k == "init" else (0, int(k.split("_")[1]))):
                batch_best = max(gens[g])
                cumulative_best = max(cumulative_best, batch_best)
                avg = sum(gens[g]) / len(gens[g])
                improved = " *" if batch_best == cumulative_best and batch_best > float("-inf") else ""
                print(f"      {g}: best={cumulative_best:.4f}  batch_best={batch_best:.4f}  avg={avg:.4f}{improved}")

    # Section 4: Local Grid winner
    lg_evals = pipeline_result.stage_results.get("LocalGridStage", [])
    if lg_evals:
        valid = [e for e in lg_evals if e.fitness_score is not None and e.fitness_score != float("-inf")]
        best_lg = max(valid, key=lambda e: e.fitness_score) if valid else None
        print(f"\n  Local Grid Stage: {len(lg_evals)} evaluated, {len(valid)} valid")
        if best_lg:
            print(f"    Best: {best_lg.fitness_score:.4f}  params={best_lg.parameters}")

    # Overall best
    if pipeline_result.best_evaluation is not None:
        print(f"\n  OVERALL BEST: {pipeline_result.best_score:.4f}")
        print(f"    params={pipeline_result.best_evaluation.parameters}")
        print(f"    stage={pipeline_result.best_evaluation.stage}")

    # Quarterly Performance Breakdown (best candidate)
    best = pipeline_result.best_evaluation
    if best is not None and best.extracted_metrics:
        trades = best.extracted_metrics.get("trades", [])
        if trades:
            from optimizer.fitness.trailing_stop import _bucket_by_quarter, _base_score

            quarters = _bucket_by_quarter(trades)
            if quarters:
                print(f"\n  {'─'*60}")
                print("  Quarterly Performance Breakdown")
                print(f"  {'─'*60}")
                print(f"  {'Quarter':>10}  {'Trades':>8}  {'Net PnL':>10}  {'PF':>8}  {'Fitness':>10}")
                print("  " + "─" * 52)

                for q_key in sorted(quarters.keys()):
                    q_trades = quarters[q_key]
                    n = len(q_trades)
                    net_pnl = sum(t.get("profit", 0) for t in q_trades)
                    gross_profit = sum(t.get("profit", 0) for t in q_trades if t.get("profit", 0) > 0)
                    gross_loss = abs(sum(t.get("profit", 0) for t in q_trades if t.get("profit", 0) < 0))
                    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
                    score = _base_score(q_trades, min_trades)
                    pf_str = f"{pf:.2f}" if pf != float("inf") else "inf"
                    print(f"  {q_key:>10}  {n:>8}  {net_pnl:>10.2f}  {pf_str:>8}  {score:>10.4f}")

    print(f"\n  Wall time: {wall_seconds:.1f}s")
    print(f"  Total evaluations: {len(pipeline_result.all_evaluations)}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_batch(args) -> None:
    raw_sets = json.loads(args.parameter_sets.read_text())
    param_sets = [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in raw_sets
    ]
    logger.info(f"Loaded {len(param_sets)} parameter sets from {args.parameter_sets}")

    results_root = _resolve_results_root()
    artifact_dir = compiler.compile(args.strategy_path, skip=args.skip_compile)
    harness_dir = compiler.compile_harness()
    logger.info(f"Artifact dir: {artifact_dir}")

    t0 = time.monotonic()
    with WorkerPool(
        n=args.worker_count,
        artifact_dir=artifact_dir,
        results_root=results_root,
        harness_dir=harness_dir,
    ) as pool:
        evaluations = run_batch(
            parameter_sets=param_sets,
            pool=pool,
            results_root=results_root,
            artifacts_container_path=ARTIFACTS_CONTAINER_PATH,
        )
    wall_seconds = time.monotonic() - t0
    _print_batch_summary(evaluations, wall_seconds)


def _parse_cli_params(items: list[str]) -> dict:
    params: dict = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid param {item!r}; expected KEY=VALUE")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid param {item!r}; empty key")
        params[key] = value
    return params


def _load_param_sets_from_file(path: Path) -> list[dict]:
    if not path.is_absolute():
        raise ValueError(f"params file must be an absolute path: {path}")
    data = json.loads(path.read_text())
    if isinstance(data, list):
        raw_sets = data
    elif isinstance(data, dict):
        raw_sets = data.get("parameter_sets", [data])
    else:
        raise ValueError("params file must be a list or dict")

    return [{k: v for k, v in p.items() if not k.startswith("_")} for p in raw_sets]


def _apply_symbol_filter(
    param_sets: list[dict],
    symbol: str,
    override_symbol: bool,
) -> list[dict]:
    filtered: list[dict] = []
    for params in param_sets:
        p = dict(params)
        if "ticker" in p:
            if override_symbol:
                p["ticker"] = symbol
            elif p["ticker"] != symbol:
                continue
        else:
            p["ticker"] = symbol
        filtered.append(p)
    return filtered


def _resolve_backtest_param_sets(
    params_kv: list[str] | None,
    params_file: Path | None,
    symbol_override: str | None = None,
    override_symbol: bool = False,
) -> list[dict]:
    if params_kv and params_file:
        raise ValueError("Provide either --param or --params-file, not both")
    if params_kv:
        param_sets = [_parse_cli_params(params_kv)]
    elif params_file:
        param_sets = _load_param_sets_from_file(params_file)
    else:
        raise ValueError("Missing parameter source")

    if symbol_override:
        param_sets = _apply_symbol_filter(param_sets, symbol_override, override_symbol)
    return param_sets


def _cmd_backtest(args) -> None:
    if args.params_file and not args.params_file.is_absolute():
        raise ValueError(f"params file must be an absolute path: {args.params_file}")
    param_sets = _resolve_backtest_param_sets(args.param, args.params_file)
    logger.info(f"Loaded {len(param_sets)} parameter sets for backtest")

    results_root = _resolve_results_root()
    BACKTESTS_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_dir = compiler.compile(args.strategy_path, skip=args.skip_compile)
    harness_dir = compiler.compile_harness()
    logger.info(f"Artifact dir: {artifact_dir}")

    t0 = time.monotonic()
    with WorkerPool(
        n=args.worker_count,
        artifact_dir=artifact_dir,
        results_root=results_root,
        harness_dir=harness_dir,
    ) as pool:
        evaluations = run_backtests(
            parameter_sets=param_sets,
            pool=pool,
            results_root=results_root,
            backtests_root=BACKTESTS_ROOT,
        )
    wall_seconds = time.monotonic() - t0
    _print_backtest_summary(evaluations, wall_seconds)


def _build_pipeline(opt_config: dict, checkpoint_path: Path, resume_path: str | None = None):
    """Build optimization pipeline from config. Returns (pipeline, space, fitness_fn, base_params, resume_from)."""
    from optimizer.fitness.registry import get_fitness
    from optimizer.pipeline.bayesian import BayesianStage
    from optimizer.pipeline.checkpoint import load_checkpoint
    from optimizer.pipeline.genetic import GeneticStage
    from optimizer.pipeline.lhs import LHSStage
    from optimizer.pipeline.local_grid import LocalGridStage
    from optimizer.pipeline.parameter_space import ParameterSpace
    from optimizer.pipeline.pipeline import OptimizationPipeline

    resume_from = None
    if resume_path:
        resume_from = load_checkpoint(Path(resume_path))
        if resume_from is None:
            raise ValueError(f"Could not load checkpoint: {resume_path}")

    # Base params (static LEAN fields merged into every parameter dict)
    base_params = opt_config.get("base_params", {})
    if base_params:
        logger.info(f"Base params: {base_params}")

    # Fitness
    fit_cfg = opt_config["fitness"]
    fitness_fn = get_fitness(
        fit_cfg["name"],
        min_trades=fit_cfg.get("min_trades", 30),
        min_profit_factor=fit_cfg.get("min_profit_factor", 1.0),
        max_drawdown_limit=fit_cfg.get("max_drawdown_limit"),
    )
    logger.info(f"Fitness function: {fit_cfg['name']}")

    # Parameter space (configurable path, fallback to default)
    ps_file = opt_config.get("parameter_space_file", "config/parameter_space.json")
    space = ParameterSpace.from_json(PROJECT_ROOT / ps_file)
    logger.info(f"Parameter space: {space.total_combinations()} total combinations ({ps_file})")

    # Assemble stages — order: LHS → Bayesian → genetic_seeded → LocalGrid
    stage_cfg = opt_config.get("stages", {})

    def _ga_kwargs(cfg: dict) -> dict:
        """Extract GeneticStage constructor kwargs from config, dropping 'enabled'."""
        return {k: v for k, v in cfg.items() if k != "enabled"}

    stages = []
    if stage_cfg.get("lhs"):
        stages.append(LHSStage(**stage_cfg["lhs"]))
    if stage_cfg.get("bayesian"):
        stages.append(BayesianStage(**stage_cfg["bayesian"]))
    ga_seeded_cfg = stage_cfg.get("genetic_seeded", {})
    if ga_seeded_cfg.get("enabled"):
        stages.append(GeneticStage(**_ga_kwargs(ga_seeded_cfg)))
        logger.info("[optimize] genetic_seeded stage enabled")
    lg_cfg = stage_cfg.get("local_grid", {})
    if lg_cfg:
        stages.append(LocalGridStage(**lg_cfg))

    # Standalone GA: replaces the main pipeline stages
    ga_standalone_cfg = stage_cfg.get("genetic_standalone", {})
    if ga_standalone_cfg.get("enabled"):
        stages = [GeneticStage(**_ga_kwargs(ga_standalone_cfg))]
        logger.info("[optimize] genetic_standalone mode — GA only, no other stages")

    pipeline = OptimizationPipeline(
        stages=stages, checkpoint_path=checkpoint_path,
    )

    return pipeline, space, fitness_fn, base_params, resume_from


def _run_single_optimize(
    opt_config: dict,
    strategy_path: Path,
    artifact_dir: Path,
    harness_dir: Path,
    results_root: Path,
    worker_count: int,
    checkpoint_path: Path,
    resume_path: str | None = None,
    container_prefix: str = "lean_optimizer",
    run_timestamp: str | None = None,
):
    """Run a single-symbol optimization. Returns (pipeline_result, wall_seconds)."""
    from optimizer.logging import EvalLogger

    pipeline, space, fitness_fn, base_params, resume_from = _build_pipeline(
        opt_config, checkpoint_path, resume_path,
    )

    # Set up per-eval JSONL logger
    ts = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    symbol = opt_config.get("base_params", {}).get("ticker", "UNKNOWN")
    tunable_names = [p.name for p in space.parameters]
    eval_logger = EvalLogger(
        base_dir=PROJECT_ROOT / "runs" / ts,
        symbol=symbol,
        tunable_params=tunable_names,
    )

    # Build batch_runner that merges base_params into every parameter dict
    def _make_batch_runner(pool, results_root, base_params):
        def batch_runner(parameter_sets: list) -> list:
            merged = [{**base_params, **ps} for ps in parameter_sets]
            return run_batch(
                parameter_sets=merged,
                pool=pool,
                results_root=results_root,
                artifacts_container_path=ARTIFACTS_CONTAINER_PATH,
            )
        return batch_runner

    t0 = time.monotonic()
    try:
        with WorkerPool(
            n=worker_count,
            artifact_dir=artifact_dir,
            results_root=results_root,
            harness_dir=harness_dir,
            container_prefix=container_prefix,
        ) as pool:
            batch_runner = _make_batch_runner(pool, results_root, base_params)
            pipeline_result = pipeline.run(
                space=space,
                fitness_fn=fitness_fn,
                batch_runner=batch_runner,
                resume_from=resume_from,
                eval_logger=eval_logger,
            )

    except KeyboardInterrupt:
        wall_seconds = time.monotonic() - t0
        logger.warning(f"\nInterrupted after {wall_seconds:.1f}s")
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        raise

    wall_seconds = time.monotonic() - t0

    return pipeline_result, wall_seconds


def _cmd_optimize(args) -> None:
    # Load optimization config
    opt_config = json.loads(args.config.read_text())
    logger.info(f"Loaded optimization config from {args.config}")

    # CLI ticker override
    if args.ticker:
        opt_config.setdefault("base_params", {})["ticker"] = args.ticker.upper()
        logger.info(f"Ticker override: {args.ticker.upper()}")

    if "ticker" not in opt_config.get("base_params", {}):
        logger.error("No ticker in config. Pass --ticker on the command line.")
        sys.exit(1)

    # Checkpoint setup
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    if args.resume:
        checkpoint_path = Path(args.resume)
    else:
        symbol = opt_config.get("base_params", {}).get("ticker", "UNKNOWN")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = logs_dir / f"checkpoint_{symbol}_{ts}.json"
    logger.info(f"Checkpoint file: {checkpoint_path}")

    # Compile
    results_root = _resolve_results_root()
    worker_count = args.worker_count or opt_config.get("worker_count", 15)
    artifact_dir = compiler.compile(args.strategy_path, skip=args.skip_compile)
    harness_dir = compiler.compile_harness()
    logger.info(f"Artifact dir: {artifact_dir}")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        pipeline_result, wall_seconds = _run_single_optimize(
            opt_config=opt_config,
            strategy_path=args.strategy_path,
            artifact_dir=artifact_dir,
            harness_dir=harness_dir,
            results_root=results_root,
            worker_count=worker_count,
            checkpoint_path=checkpoint_path,
            resume_path=args.resume,
            run_timestamp=run_timestamp,
        )
    except KeyboardInterrupt:
        logger.info(f"Resume with: python main.py optimize --config {args.config} --resume {checkpoint_path}")
        return

    if pipeline_result.interrupted:
        logger.info(f"Checkpoint saved to: {checkpoint_path}")
        logger.info(f"Resume with: python main.py optimize --config {args.config} --resume {checkpoint_path}")

    fit_cfg = opt_config.get("fitness", {})
    _print_optimize_summary(pipeline_result, wall_seconds, min_trades=fit_cfg.get("min_trades", 100))


# ---------------------------------------------------------------------------
# Universe (multi-symbol) mode
# ---------------------------------------------------------------------------


def _run_symbol_worker(
    symbol: str,
    base_config: dict,
    strategy_path: str,
    artifact_dir: str,
    harness_dir: str,
    results_root: str,
    workers_per_symbol: int,
    timestamp: str,
) -> dict:
    """Run optimization for a single symbol. Called in a subprocess via ProcessPoolExecutor.

    All arguments are plain strings/dicts/ints (picklable). Returns a summary dict.
    """
    # Ignore SIGINT in workers — let the main process handle it
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Per-symbol log file
    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"universe_{symbol}_{timestamp}.log"
    logger.add(str(log_file), level="DEBUG", filter=None, enqueue=False)
    logger.info(f"[{symbol}] Starting optimization ({workers_per_symbol} workers)")

    # Override ticker in config
    config = json.loads(json.dumps(base_config))  # deep copy
    config.setdefault("base_params", {})["ticker"] = symbol

    checkpoint_path = logs_dir / f"checkpoint_{symbol}_{timestamp}.json"

    try:
        pipeline_result, wall_seconds = _run_single_optimize(
            opt_config=config,
            strategy_path=Path(strategy_path),
            artifact_dir=Path(artifact_dir),
            harness_dir=Path(harness_dir),
            results_root=Path(results_root),
            worker_count=workers_per_symbol,
            checkpoint_path=checkpoint_path,
            container_prefix=f"lean_optimizer_{symbol}",
            run_timestamp=timestamp,
        )

        # Build summary
        best_fitness = None
        best_params = None
        eval_count = len(pipeline_result.all_evaluations)
        if pipeline_result.best_evaluation is not None:
            best_fitness = pipeline_result.best_score
            best_params = pipeline_result.best_evaluation.parameters

        logger.info(f"[{symbol}] Done in {wall_seconds:.1f}s — best={best_fitness}")
        return {
            "symbol": symbol,
            "best_fitness": best_fitness,
            "best_params": best_params,
            "wall_time": wall_seconds,
            "eval_count": eval_count,
            "error": None,
        }

    except Exception as exc:
        logger.error(f"[{symbol}] Failed: {exc}")
        return {
            "symbol": symbol,
            "best_fitness": None,
            "best_params": None,
            "wall_time": None,
            "eval_count": 0,
            "error": str(exc),
        }


def _print_universe_summary(results: list[dict], wall_seconds: float) -> None:
    print(f"\n{'='*90}")
    print("  UNIVERSE OPTIMIZATION RESULTS")
    print(f"{'='*90}")

    header = f"  {'Symbol':<8} {'Best Fitness':>13} {'Wall Time':>10} {'Evals':>6}  Status"
    print(f"\n{header}")
    print("  " + "-" * 70)

    for r in sorted(results, key=lambda x: x["symbol"]):
        sym = r["symbol"]
        bf = f"{r['best_fitness']:.4f}" if r["best_fitness"] is not None else "n/a"
        wt = f"{r['wall_time']:.1f}s" if r["wall_time"] is not None else "n/a"
        ec = str(r["eval_count"])
        status = r["error"] if r["error"] else "ok"
        print(f"  {sym:<8} {bf:>13} {wt:>10} {ec:>6}  {status}")

    print("  " + "-" * 80)
    ok = sum(1 for r in results if r["error"] is None)
    print(f"  {ok}/{len(results)} symbols completed")
    print(f"  Total wall time: {wall_seconds:.1f}s")
    print(f"{'='*90}\n")


def _cmd_optimize_universe(args) -> None:
    # Load universe config
    universe_config = json.loads(args.universe.read_text())
    symbols = universe_config["symbols"]
    base_config_path = PROJECT_ROOT / universe_config["base_config"]
    symbols_parallel = universe_config.get("symbols_parallel", 3)
    total_workers = universe_config.get("total_workers", 18)

    # CLI overrides
    if args.worker_count:
        total_workers = args.worker_count

    workers_per_symbol = total_workers // symbols_parallel

    logger.info(
        f"[universe] {len(symbols)} symbols, {symbols_parallel} parallel, "
        f"{total_workers} total workers ({workers_per_symbol} per symbol)"
    )

    # Load base config template
    base_config = json.loads(base_config_path.read_text())

    # Compile once in main process
    results_root = _resolve_results_root()
    artifact_dir = compiler.compile(args.strategy_path, skip=args.skip_compile)
    harness_dir = compiler.compile_harness()
    logger.info(f"Artifact dir: {artifact_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    t0 = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=symbols_parallel) as executor:
            futures = {
                executor.submit(
                    _run_symbol_worker,
                    symbol=sym,
                    base_config=base_config,
                    strategy_path=str(args.strategy_path),
                    artifact_dir=str(artifact_dir),
                    harness_dir=str(harness_dir),
                    results_root=str(results_root),
                    workers_per_symbol=workers_per_symbol,
                    timestamp=timestamp,
                ): sym
                for sym in symbols
            }

            for future in as_completed(futures):
                sym = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    bf = f"{result['best_fitness']:.4f}" if result["best_fitness"] is not None else "FAILED"
                    wt = f"{result['wall_time']:.1f}s" if result["wall_time"] is not None else "n/a"
                    logger.info(f"[universe] {sym} done in {wt} — best={bf}")
                except Exception as exc:
                    logger.error(f"[universe] {sym} raised: {exc}")
                    results.append({
                        "symbol": sym, "best_fitness": None, "best_params": None,
                        "wall_time": None, "eval_count": 0,
                        "error": str(exc),
                    })

    except KeyboardInterrupt:
        wall_seconds = time.monotonic() - t0
        logger.warning(f"\n[universe] Interrupted after {wall_seconds:.1f}s")
        logger.info("[universe] Waiting for running symbols to clean up...")
        # ProcessPoolExecutor.__exit__ handles shutdown; workers ignore SIGINT
        # and clean up via WorkerPool context manager
        if results:
            _print_universe_summary(results, wall_seconds)
        return

    wall_seconds = time.monotonic() - t0
    _print_universe_summary(results, wall_seconds)


def _run_backtest_symbol_worker(
    symbol: str,
    param_sets: list[dict],
    strategy_path: str,
    artifact_dir: str,
    harness_dir: str,
    results_root: str,
    backtests_root: str,
    workers_per_symbol: int,
    timestamp: str,
) -> dict:
    """Run backtests for a single symbol (subprocess via ProcessPoolExecutor)."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logs_dir = PROJECT_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"backtest_{symbol}_{timestamp}.log"
    logger.add(str(log_file), level="DEBUG", filter=None, enqueue=False)
    logger.info(f"[{symbol}] Starting backtests ({workers_per_symbol} workers)")

    if not param_sets:
        return {
            "symbol": symbol,
            "count": 0,
            "wall_time": None,
            "error": "no parameter sets for symbol",
        }

    t0 = time.monotonic()
    try:
        with WorkerPool(
            n=workers_per_symbol,
            artifact_dir=Path(artifact_dir),
            results_root=Path(results_root),
            harness_dir=Path(harness_dir),
            container_prefix=f"lean_backtest_{symbol}",
        ) as pool:
            run_backtests(
                parameter_sets=param_sets,
                pool=pool,
                results_root=Path(results_root),
                backtests_root=Path(backtests_root),
            )
    except Exception as exc:
        logger.error(f"[{symbol}] Failed: {exc}")
        return {
            "symbol": symbol,
            "count": 0,
            "wall_time": None,
            "error": str(exc),
        }

    wall_seconds = time.monotonic() - t0
    logger.info(f"[{symbol}] Done in {wall_seconds:.1f}s")
    return {
        "symbol": symbol,
        "count": len(param_sets),
        "wall_time": wall_seconds,
        "error": None,
    }


def _print_backtest_universe_summary(results: list[dict], wall_seconds: float) -> None:
    print(f"\n{'='*80}")
    print("  UNIVERSE BACKTEST RESULTS")
    print(f"{'='*80}")

    header = f"  {'Symbol':<8} {'Backtests':>10} {'Wall Time':>10}  Status"
    print(f"\n{header}")
    print("  " + "-" * 60)

    for r in sorted(results, key=lambda x: x["symbol"]):
        sym = r["symbol"]
        count = str(r["count"])
        wt = f"{r['wall_time']:.1f}s" if r["wall_time"] is not None else "n/a"
        status = r["error"] if r["error"] else "ok"
        print(f"  {sym:<8} {count:>10} {wt:>10}  {status}")

    print("  " + "-" * 60)
    ok = sum(1 for r in results if r["error"] is None)
    print(f"  {ok}/{len(results)} symbols completed")
    print(f"  Total wall time: {wall_seconds:.1f}s")
    print(f"{'='*80}\n")


def _cmd_backtest_universe(args) -> None:
    universe_config = json.loads(args.universe.read_text())
    symbols = universe_config["symbols"]
    symbols_parallel = universe_config.get("symbols_parallel", 3)
    total_workers = universe_config.get("total_workers", 18)

    param_sets_by_symbol = universe_config.get("param_sets", {})
    if not isinstance(param_sets_by_symbol, dict):
        raise ValueError("universe config 'param_sets' must be a dict keyed by symbol")

    if args.worker_count:
        total_workers = args.worker_count
    else:
        total_workers = min(len(symbols), 15)

    workers_per_symbol = max(1, total_workers // symbols_parallel)
    logger.info(
        f"[backtest-universe] {len(symbols)} symbols, {symbols_parallel} parallel, "
        f"{total_workers} total workers ({workers_per_symbol} per symbol)"
    )

    results_root = _resolve_results_root()
    BACKTESTS_ROOT.mkdir(parents=True, exist_ok=True)
    artifact_dir = compiler.compile(args.strategy_path, skip=args.skip_compile)
    harness_dir = compiler.compile_harness()
    logger.info(f"Artifact dir: {artifact_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results: list[dict] = []

    t0 = time.monotonic()
    try:
        with ProcessPoolExecutor(max_workers=symbols_parallel) as executor:
            futures = {
                executor.submit(
                    _run_backtest_symbol_worker,
                    symbol=sym,
                    param_sets=_apply_symbol_filter(
                        param_sets_by_symbol.get(sym, []),
                        sym,
                        override_symbol=True,
                    ),
                    strategy_path=str(args.strategy_path),
                    artifact_dir=str(artifact_dir),
                    harness_dir=str(harness_dir),
                    results_root=str(results_root),
                    backtests_root=str(BACKTESTS_ROOT),
                    workers_per_symbol=workers_per_symbol,
                    timestamp=timestamp,
                ): sym
                for sym in symbols
            }

            for future in as_completed(futures):
                sym = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    count = result["count"]
                    wt = f"{result['wall_time']:.1f}s" if result["wall_time"] is not None else "n/a"
                    logger.info(f"[backtest-universe] {sym} done in {wt} — {count} backtests")
                except Exception as exc:
                    logger.error(f"[backtest-universe] {sym} raised: {exc}")
                    results.append({
                        "symbol": sym,
                        "count": 0,
                        "wall_time": None,
                        "error": str(exc),
                    })

    except KeyboardInterrupt:
        wall_seconds = time.monotonic() - t0
        logger.warning(f"\n[backtest-universe] Interrupted after {wall_seconds:.1f}s")
        if results:
            _print_backtest_universe_summary(results, wall_seconds)
        return

    wall_seconds = time.monotonic() - t0
    _print_backtest_universe_summary(results, wall_seconds)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="lean-optimizer — parallel batch runner & optimization pipeline"
    )
    subparsers = parser.add_subparsers(dest="mode", help="Execution mode")

    # Shared arguments
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--strategy-path",
        type=Path,
        default=DEFAULT_STRATEGY,
        help="Path to C# strategy project directory",
    )
    shared.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip compilation (reuse cached artifact if present)",
    )
    shared.add_argument(
        "--worker-count",
        type=int,
        default=int(os.environ.get("WORKER_COUNT", "0")),
        help="Number of parallel worker containers",
    )

    # batch subcommand
    batch_parser = subparsers.add_parser(
        "batch", parents=[shared], help="Run a fixed set of parameter dicts"
    )
    batch_parser.add_argument(
        "--parameter-sets",
        type=Path,
        default=DEFAULT_PARAM_SETS,
        help="Path to JSON file containing list of parameter dicts",
    )

    # optimize subcommand
    opt_parser = subparsers.add_parser(
        "optimize", parents=[shared], help="Run full optimization pipeline"
    )
    opt_parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_OPT_CONFIG,
        help="Path to optimization config JSON",
    )
    opt_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Override ticker in config (avoids per-symbol config files)",
    )
    opt_parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint file to resume from",
    )

    # optimize-universe subcommand
    univ_parser = subparsers.add_parser(
        "optimize-universe", parents=[shared],
        help="Optimize multiple symbols in parallel",
    )
    univ_parser.add_argument(
        "--universe",
        type=Path,
        required=True,
        help="Path to universe config JSON (symbols, base_config, symbols_parallel, total_workers)",
    )

    # backtest subcommand
    backtest_parser = subparsers.add_parser(
        "backtest", parents=[shared], help="Run one or more backtests"
    )
    bt_group = backtest_parser.add_mutually_exclusive_group(required=True)
    bt_group.add_argument(
        "--param",
        action="append",
        help="Parameter override in KEY=VALUE form (repeatable)",
    )
    bt_group.add_argument(
        "--params-file",
        type=Path,
        help="Absolute path to JSON params file (list or dict)",
    )

    # backtest-universe subcommand
    bt_univ_parser = subparsers.add_parser(
        "backtest-universe", parents=[shared],
        help="Run backtests across multiple symbols in parallel",
    )
    bt_univ_parser.add_argument(
        "--universe",
        type=Path,
        required=True,
        help="Path to universe config JSON (symbols, param_sets, symbols_parallel, total_workers)",
    )

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        return

    worker_count_provided = "--worker-count" in sys.argv
    if args.mode in {"backtest", "backtest-universe"} and not worker_count_provided:
        args.worker_count = 0

    # Default worker count for batch/backtest modes
    if args.worker_count == 0 and args.mode == "batch":
        args.worker_count = int(os.environ.get("WORKER_COUNT", "5"))
    if args.worker_count == 0 and args.mode == "backtest":
        args.worker_count = 1

    if args.mode == "batch":
        _cmd_batch(args)
    elif args.mode == "backtest":
        _cmd_backtest(args)
    elif args.mode == "optimize":
        _cmd_optimize(args)
    elif args.mode == "optimize-universe":
        _cmd_optimize_universe(args)
    elif args.mode == "backtest-universe":
        _cmd_backtest_universe(args)


if __name__ == "__main__":
    main()
