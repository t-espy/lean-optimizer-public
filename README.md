# lean-optimizer — Parallel Optimization Engine for QuantConnect LEAN

Production-grade parallel optimization engine for QuantConnect LEAN algorithmic trading strategies. Compiles a C# strategy once, spins up a pool of warm Docker containers with persistent .NET harness processes, and searches the parameter space via a multi-stage pipeline — reducing a 679,140-point parameter search from ~88 hours to under 3 minutes. 177 passing tests.

## Performance

Measured on a 7-parameter space (679,140 grid points), 15 parallel workers, 20-core machine. Both runs use the same strategy, parameter ranges, date range, and data.

### lean-optimizer vs LEAN CLI grid search

| | LEAN CLI (`lean optimize --strategy "grid search"`) | lean-optimizer (GA standalone) |
|---|---|---|
| **Search strategy** | Exhaustive grid (all 679,140 points) | Incremental genetic algorithm |
| **Evaluations** | 679,140 | 611 |
| **Throughput (15 workers)** | 2.15 evals/sec | 3.82 evals/sec |
| **Measured wall time** | 1,772 evals in 825s (extrapolated: ~88 hours for full grid) | **2m 40s** |
| **Search efficiency** | 1x | **1,112x fewer evaluations** |

The two advantages are independent and compound:

1. **Persistent harness** — eliminates .NET startup and LEAN engine init overhead per evaluation, yielding ~1.8x higher throughput at the container level.
2. **Smart search** — the GA converged in 611 evaluations (early-stopped at generation 42) instead of exhaustively evaluating all 679,140 grid points — **1,112x fewer evaluations**.

Combined: what takes LEAN CLI's grid search an estimated **~88 hours** completes in **under 3 minutes**.

## Architecture

The default configuration uses a **standalone incremental genetic algorithm** — the same search strategy benchmarked above. The GA uses tournament selection, worst-replacement, and early stopping, converging on an optimum without exhaustive evaluation.

Additional pipeline stages are available and can be enabled independently via config:

- **Latin Hypercube Sampling (LHS)** — space-filling initial exploration
- **Bayesian Optimization (Optuna TPE)** — directed search with batch ask/tell
- **Local Grid Search** — Chebyshev-distance neighborhood refinement around top candidates

All stages share a common `BatchRunner` interface and deduplication cache. The GA can also run seeded from prior stages (LHS/Bayesian results injected into the initial population).

Walk-forward optimization (true IS/OOS parameter validation) is a planned future stage. The current pipeline produces optimized parameters with checkpoint/resume support; out-of-sample validation is left to the user's preferred methodology.

### Persistent Harness

The core innovation: each Docker container runs a long-lived `dotnet /Harness/LeanHarness.dll` process. The harness reads JSON requests from stdin, runs backtests with full LEAN state reset between runs, and writes JSON responses to stdout. This eliminates per-eval .NET startup, assembly loading, and LEAN engine initialization overhead — measured at ~1.8x higher throughput per worker compared to LEAN CLI's per-backtest container execution.

```
Python → harness stdin:  {"id": "<uuid>", "config_path": "/path/to/config.json"}
Harness → Python stdout: {"id": "<uuid>", "status": "complete"}
```

`Console.Out` is redirected to stderr inside the harness so LEAN logging doesn't corrupt the JSON protocol. On eval error, the harness logs the error and continues — it does not exit. On stdin EOF, it exits cleanly.

## Requirements

**Infrastructure you provide:**

- **Docker** with a LEAN engine image (e.g., `quantconnect/lean:latest` or a custom build). Set `LEAN_IMAGE` in `.env`.
- **QuantConnect LEAN source** — a local clone with pre-built DLLs. **The harness build will fail if this is not configured correctly.** The `.csproj` at `harness/LeanHarness/LeanHarness.csproj` references LEAN DLLs via relative `HintPath` entries (default: `../../../LeanCode/Lean/Launcher/bin/Debug/`). You must either:
  - Clone this repo so the relative path resolves to your LEAN install, **or**
  - Edit every `HintPath` in the `.csproj` to point to your LEAN DLL location, **or**
  - Create a symlink: `ln -s /path/to/your/LeanCode /path/to/lean-optimizer/../LeanCode`
- **Market data** — LEAN-format data on a fast filesystem. A tmpfs/RAM disk is strongly recommended for parallel workloads. Set `LEAN_DATA_HOST_PATH` in `.env`. **This directory must contain both your historical price data AND LEAN's reference data files.** LEAN requires several infrastructure files at runtime — without them, backtests will silently produce empty results. At minimum, copy these from your LEAN source (`Lean/Data/`):
  ```
  symbol-properties/symbol-properties-database.csv
  market-hours/market-hours-database.json
  ```
  The safest approach is to copy the entire `Lean/Data/` tree to your data path and then add your price data under `equity/usa/minute/` (or the appropriate resolution/market). If you use a tmpfs/RAM disk, remember that **its contents are lost on reboot or crash** — you will need to repopulate it.
- **A C# strategy project** — a directory containing `.cs` and `.csproj` files that compile against QuantConnect. Pass via `--strategy-path`.
- **Python 3.12+**
- **.NET 8.0+** — for compiling the harness and your strategy
- A multi-core machine with sufficient RAM for parallel workers (tested on 20-core/128GB)

**Setup:**

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # edit all paths for your environment
```

**Environment variables** (in `.env` — see `.env.example`):

| Variable | Purpose |
|----------|---------|
| `LEAN_IMAGE` | Docker image for LEAN engine containers |
| `LEAN_DATA_HOST_PATH` | Host path to LEAN-format market data (mounted read-only into containers) |
| `LEAN_WORKSPACE` | Host path to your LEAN workspace (for algorithm class auto-detection) |
| `RESULTS_SCRATCH_PATH` | Scratch directory for per-eval results (default: `./results`) |
| `WORKER_COUNT` | Default parallel worker count (overridden by `--worker-count` or config) |

## Usage

```bash
# Optimize a strategy (full pipeline)
python main.py optimize --strategy-path /path/to/your_strategy --config config/optimization_ga.json --ticker AAPL

# Multi-symbol universe optimization
python main.py optimize-universe --strategy-path /path/to/your_strategy --universe config/universe.json
```

**Key flags:**
- `--strategy-path PATH` — directory containing your C# strategy `.cs` and `.csproj` files
- `--config PATH` — optimization config (fitness function, stage params, worker count)
- `--ticker SYMBOL` — override `base_params.ticker` from CLI (eliminates per-symbol config files)
- `--worker-count N` — parallel Docker containers (default from config or env)
- `--resume` — resume from checkpoint (skips completed stages)
- `--skip-compile` — skip strategy compilation (use cached artifacts)

### Batch Backtesting

A batch backtest runner is included as a subcommand for running portfolio-wide or multi-symbol backtests independently of the optimization pipeline. It uses the same warm worker pool and persistent harness as the optimizer.

```bash
# Single symbol, explicit parameters
python main.py backtest --strategy-path /path/to/your_strategy --param ticker=AAPL --param startDate=20250101 --param endDate=20251231

# Multi-symbol backtests from a universe config
python main.py backtest-universe --strategy-path /path/to/your_strategy --universe config/backtest_universe.json
```

## Configuration

**`config/optimization_ga.json`** — optimization config. The included file has example `base_params` (dates, cash, etc.) that you should adjust for your strategy and data range:

| Key | Default | Controls |
|-----|---------|----------|
| `fitness.name` | `"calmar"` | Fitness function (registry lookup) |
| `fitness.min_trades` | `10` | Minimum trade count for valid score |
| `fitness.min_profit_factor` | `1.0` | Minimum profit factor for valid score |
| `stages.genetic_standalone.population_size` | `125` | GA population size |
| `stages.genetic_standalone.n_generations` | `73` | Max GA generations |
| `stages.genetic_standalone.batch_size` | `15` | Children evaluated per GA generation |
| `stages.genetic_standalone.early_stopping_generations` | `15` | Plateau window for early stop |
| `stages.genetic_standalone.early_stopping_min_delta` | `0.01` | Min improvement to avoid early stop |
| `stages.genetic_standalone.crossover_prob` | `0.95` | Crossover probability |
| `stages.genetic_standalone.mutation_prob` | `0.05` | Per-parameter mutation probability |
| `stages.genetic_standalone.tournament_size` | `3` | Tournament selection pool size |
| `worker_count` | `15` | Parallel Docker worker containers |

Additional stage config keys (when enabled): `stages.lhs.n_samples`, `stages.bayesian.n_calls`, `stages.bayesian.batch_size`, `stages.local_grid.top_n`, `stages.local_grid.radius`, `stages.local_grid.max_neighbors`.

**`config/parameter_space.json`** — defines tunable parameters with min, max, step, and type (int/float). The optimizer generates all valid grid points and uses them for LHS, GA mutation, and local grid neighbor generation.

**The included parameter space is a placeholder example** (3 generic parameters, 180 combinations). You must replace it with parameters that match your strategy's actual `GetParameter()` calls — otherwise the optimizer will run but the parameters won't affect backtest behavior, and results will be meaningless. Each entry's `name` must exactly match a `GetParameter()` key in your C# strategy:

```csharp
// In your strategy's Initialize():
var lookback = GetParameter("MyLookback", 20);  // ← name must match
var threshold = GetParameter("MyThreshold", 0.5);
```

```json
// In parameter_space.json:
{
  "parameters": [
    { "name": "MyLookback", "min": 10, "max": 50, "step": 5, "type": "int" },
    { "name": "MyThreshold", "min": 0.1, "max": 0.9, "step": 0.1, "type": "float" }
  ]
}
```

The optimizer passes these values via the LEAN config's `parameters` block. If a `name` doesn't match any `GetParameter()` call, LEAN silently ignores it and your strategy uses its hardcoded default.

**Environment variables:** `LEAN_IMAGE`, `LEAN_DATA_HOST_PATH`, `LEAN_WORKSPACE`, `ARTIFACTS_CONTAINER_PATH`, `RESULTS_SCRATCH_PATH`, `WORKER_COUNT`. See `.env.example`.

## Architecture Decisions

These constraints are load-bearing. Respect them when extending.

1. **Compile-once.** `optimizer/builder/compiler.py` hashes strategy source files. If the hash matches a cached build, compilation is skipped. Never compile per-evaluation.

2. **Warm workers.** `WorkerPool` starts N long-lived Docker containers at init. Workers are acquired/released via a blocking queue — never create/destroy containers per backtest.

3. **Persistent harness.** Each Docker container runs a long-lived `dotnet /Harness/LeanHarness.dll` process launched at container start via `subprocess.Popen(["docker", "exec", "-i", ...])`. The harness reads JSON requests from stdin, runs backtests with full LEAN state reset between runs, and writes JSON responses to stdout. This eliminates per-eval .NET startup, assembly loading, and LEAN engine init overhead (~1.8x throughput improvement vs LEAN CLI's per-backtest execution). The harness is compiled on the host and mounted read-only at `/Harness` — the Docker image is NOT modified.

4. **Harness protocol.** Python → harness stdin: `{"id": "<uuid>", "config_path": "/path/to/config.json"}`. Harness → Python stdout: `{"id": "<uuid>", "status": "complete"}` or `{"id": "<uuid>", "status": "error", "message": "..."}`. `Console.Out` is redirected to stderr inside the harness so LEAN logging doesn't corrupt the JSON protocol. On eval error, the harness logs the error and continues the loop — it does not exit. On stdin EOF, the harness exits cleanly.

5. **RAM drive.** Market data should live on a fast filesystem (tmpfs recommended). LEAN containers mount this read-only. Never copy data into containers. The data directory must include LEAN's reference data (`symbol-properties/`, `market-hours/`, etc.) alongside your price data — LEAN will fail silently without them. tmpfs is volatile: contents are lost on reboot/crash.

6. **Injected stages.** All pipeline stages implement `OptimizationStage` (see `optimizer/pipeline/base.py`). Stages receive a `BatchRunner` callable — they never touch Docker, workers, or compilation directly. New stages just implement `run(space, fitness_fn, batch_runner, previous_results)`.

7. **Fitness registry.** `optimizer/fitness/registry.py` maps string names to fitness classes. Config specifies `"name": "calmar"`, the registry instantiates it. Add new fitness functions by subclassing `FitnessFunction` and registering.

8. **Incremental GA.** `GeneticStage` uses worst-replacement (not generational). Children only enter the population if they beat the current worst member. Best fitness is monotonically non-decreasing. Early stopping exits when best hasn't improved by `min_delta` in N generations.

9. **Checkpoint after every stage.** Pipeline saves atomic JSON checkpoints after each stage completes. Resume skips completed stages by positional+name match.

10. **Per-eval JSONL logging.** `EvalLogger` writes one JSON line per evaluation to `runs/<timestamp>/<symbol>/evaluations.jsonl`. File is opened/closed per batch (no persistent handle). Params are filtered to tunable-only (excludes base_params like ticker, startDate, endDate). Shared timestamp across symbols in universe mode.

## Project Structure

```
main.py                                  CLI entry point: batch/optimize/backtest subcommands
config/base_config.json                  LEAN backtesting config template
config/optimization_ga.json              Optimization config (fitness, stage params, worker count)
config/parameter_space.json              Tunable parameter definitions (example: 3 params)

harness/LeanHarness/Program.cs           Persistent LEAN harness: stdin/stdout JSON protocol, full state reset
harness/LeanHarness/LeanHarness.csproj   Harness project file, references pre-built LEAN DLLs

optimizer/fitness/base.py                Abstract FitnessFunction base class with score/compute
optimizer/fitness/calmar.py              Calmar fitness: (net_pnl / max_drawdown) * log-trade penalty
optimizer/fitness/registry.py            String-name -> fitness-class registry

optimizer/pipeline/base.py               OptimizationStage ABC + BatchRunner protocol
optimizer/pipeline/parameter_space.py    Parameter/ParameterSpace: grid values, LHS, neighbors, snap
optimizer/pipeline/scoring.py            score_evaluation(): attach fitness score to Evaluation in-place
optimizer/pipeline/lhs.py                LHS stage: space-filling initial exploration
optimizer/pipeline/bayesian.py           Bayesian stage: Optuna TPE with batch ask/tell
optimizer/pipeline/genetic.py            Incremental GA: tournament select, worst replacement, early stop
optimizer/pipeline/local_grid.py         Local grid: neighbors around top-N from prior stages
optimizer/pipeline/checkpoint.py         Atomic JSON checkpoint save/load for pipeline resume
optimizer/pipeline/pipeline.py           OptimizationPipeline orchestrator: sequential stages + checkpoint

optimizer/runner/evaluation.py           Evaluation dataclass: params, status, metrics, fitness, worker_id
optimizer/runner/batch_runner.py         Parallel batch dispatch to WorkerPool via ThreadPoolExecutor
optimizer/runner/backtest_runner.py      Backtest runner: copy full LEAN result JSON to backtests/

optimizer/results/extractor.py           Parse LEAN result JSON -> ExtractedMetrics dataclass
optimizer/results/collector.py           Verify result JSON exists, extract metrics, clean scratch dir

optimizer/builder/compiler.py            Compile C# strategy in Docker with content-hash caching + harness build

optimizer/docker/worker.py               Worker: persistent harness subprocess via stdin/stdout JSON
optimizer/docker/pool.py                 WorkerPool: N warm containers with acquire/release queue

optimizer/logging/eval_logger.py         Per-eval JSONL logger: writes fitness, metrics, params per evaluation
optimizer/config/config_builder.py       Merge base LEAN config with per-eval parameter overrides

scripts/ga_early_stop_analysis.py        GA convergence analysis: identify peak generation + plateau

tests/                                   153 passing tests covering all modules
```

## Development Notes

### Incremental GA replaces generational GA
The old generational model (with elitism) was collapsing populations by gen ~26. The incremental model uses worst replacement — children only enter if they beat the current worst. Best fitness is monotonically non-decreasing. `_fingerprint()` provides O(1) deduplication. `stage_detail` for GA init batch is `"init"`, not `"gen_0"`.

### Persistent harness design
The harness uses LEAN's regression test reset sequence: `Config.Reset()`, `Composer.Instance.Reset()`, `SymbolCache.Clear()`, etc. — the same sequence LEAN's own test suite uses to run multiple backtests in one process. `Console.Out` redirect to stderr is critical — LEAN writes trace/debug output to Console.Out. A fresh `HarnessWorkerThread` is created per run because `Engine.Run()` disposes the worker thread. The `composer-dll-directory` config must point to the LEAN launcher bin directory inside the container for MEF assembly scanning.

### NullLogHandler
The harness implements `ILogHandler` with empty Trace/Debug/Error/Dispose methods. Set via `HARNESS_DEBUG=0` (default). With `HARNESS_DEBUG=1`, falls back to `ConsoleErrorLogHandler` for troubleshooting. Testing showed LEAN logging is NOT a performance bottleneck — suppressing all output produced no measurable improvement. Per-eval time is dominated by actual backtest computation.

### Server GC rejected
`DOTNET_gcServer=1` with `DOTNET_GCHeapCount=2` added 26s wall time. Workstation GC (default) wins for parallel single-threaded workers competing for cores — Server GC's parallel collection threads add contention.

### Release build
Strategy DLLs and harness compile with `Configuration=Release`. JIT cost is amortized across evals per worker, so R2R is unnecessary.

## License

MIT

## Disclaimer

This software is for research and educational purposes. It is not financial advice. Use in live trading is at your own risk.  Benchmark results reflect specific hardware and configuration conditions. Performance will vary. The author is not liable for any losses or damages arising from the use of this software.