import optuna
from loguru import logger

from optimizer.fitness.base import FitnessFunction
from optimizer.pipeline.base import BatchRunner, OptimizationStage
from optimizer.pipeline.parameter_space import ParameterSpace
from optimizer.pipeline.scoring import score_evaluation

INF_SUBSTITUTE = -1000.0


class BayesianStage(OptimizationStage):
    """Bayesian optimization stage using optuna TPESampler.

    Uses batch ask/tell: asks for batch_size points at once, runs them
    in parallel via batch_runner, then tells all results back to the study.
    """

    def __init__(
        self,
        n_calls: int = 250,
        batch_size: int = 15,
        random_state: int | None = None,
    ):
        self.n_calls = n_calls
        self.batch_size = batch_size
        self.random_state = random_state

    def run(
        self,
        space: ParameterSpace,
        fitness_fn: FitnessFunction,
        batch_runner: BatchRunner,
        previous_results: list | None = None,
    ) -> list:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        distributions = space.to_optuna_distributions()

        # Warm-start with previous results
        tunable_names = {p.name for p in space.parameters}
        if previous_results:
            n_warm = 0
            for ev in previous_results:
                if ev.fitness_score is not None:
                    safe_score = (
                        ev.fitness_score
                        if ev.fitness_score != float("-inf")
                        else INF_SUBSTITUTE
                    )
                    tunable_params = {
                        k: v for k, v in ev.parameters.items() if k in tunable_names
                    }
                    trial = optuna.trial.create_trial(
                        params=tunable_params,
                        distributions=distributions,
                        values=[safe_score],
                    )
                    study.add_trial(trial)
                    n_warm += 1
            logger.info(f"[BayesianStage] Warm-started with {n_warm} prior results")

        new_evals = []
        best_score = float("-inf")
        best_params = None
        completed = 0

        while completed < self.n_calls:
            # Ask for a batch of points
            batch_n = min(self.batch_size, self.n_calls - completed)
            trials = [study.ask(distributions) for _ in range(batch_n)]
            param_sets = [
                {p.name: p.snap(t.params[p.name]) for p in space.parameters}
                for t in trials
            ]

            # Run batch in parallel
            evals = batch_runner(param_sets)

            # Score and tell
            for trial, ev in zip(trials, evals):
                ev.stage = "bayesian"
                score_evaluation(ev, fitness_fn)

                safe_score = (
                    ev.fitness_score
                    if ev.fitness_score != float("-inf")
                    else INF_SUBSTITUTE
                )
                study.tell(trial, safe_score)

                if ev.fitness_score > best_score:
                    best_score = ev.fitness_score
                    best_params = ev.parameters

                new_evals.append(ev)

            completed += batch_n

            logger.info(
                f"[BayesianStage] {completed}/{self.n_calls} done"
                f"  best={best_score:.4f}"
            )

        logger.info(
            f"[BayesianStage] Complete. {len(new_evals)} evaluations."
            f"  best={best_score:.4f}"
        )
        return new_evals
