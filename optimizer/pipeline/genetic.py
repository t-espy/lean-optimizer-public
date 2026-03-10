import random

from loguru import logger

from optimizer.fitness.base import FitnessFunction
from optimizer.pipeline.base import BatchRunner, OptimizationStage
from optimizer.pipeline.parameter_space import ParameterSpace
from optimizer.pipeline.scoring import score_evaluation


def _fingerprint(params: dict, keys: set) -> frozenset:
    return frozenset((k, v) for k, v in params.items() if k in keys)


def _random_individual(space: ParameterSpace, rng: random.Random) -> dict:
    """Generate a random valid grid point."""
    return {
        p.name: rng.choice(p.valid_values()) for p in space.parameters
    }


class GeneticStage(OptimizationStage):
    """Incremental genetic algorithm with worst replacement.

    Children only enter the population if they beat the current worst member.
    Good solutions are never evicted — fitness is monotonically non-decreasing.
    """

    def __init__(
        self,
        population_size: int = 125,
        n_generations: int = 73,
        crossover_prob: float = 0.95,
        mutation_prob: float = 0.05,
        tournament_size: int = 3,
        batch_size: int = 15,
        early_stopping_generations: int = 15,
        early_stopping_min_delta: float = 0.01,
        seed_from_previous: bool = False,
        n_seeds: int = 10,
        random_state: int | None = None,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.batch_size = batch_size
        self.early_stopping_generations = early_stopping_generations
        self.early_stopping_min_delta = early_stopping_min_delta
        self.seed_from_previous = seed_from_previous
        self.n_seeds = n_seeds
        self.random_state = random_state

    def run(
        self,
        space: ParameterSpace,
        fitness_fn: FitnessFunction,
        batch_runner: BatchRunner,
        previous_results: list | None = None,
    ) -> list:
        rng = random.Random(self.random_state)
        tunable_names = {p.name for p in space.parameters}
        all_evals = []

        # --- 1. Init population (list of param dicts) ---
        init_individuals = self._init_population(
            space, previous_results, rng, tunable_names,
        )

        # --- 2. Evaluate entire initial population ---
        eval_cache: dict[frozenset, object] = {}
        novel_params = []
        for ind in init_individuals:
            fp = _fingerprint(ind, tunable_names)
            if fp not in eval_cache:
                novel_params.append(ind)
                eval_cache[fp] = None  # placeholder to dedup within batch

        if novel_params:
            evals = batch_runner(novel_params)
            for j, ev in enumerate(evals):
                ev.stage = "genetic"
                ev.stage_detail = "init"
                score_evaluation(ev, fitness_fn)
                fp = _fingerprint(novel_params[j], tunable_names)
                eval_cache[fp] = ev
                all_evals.append(ev)

        # Build population as list[(params, fitness)]
        population: list[tuple[dict, float]] = []
        for ind in init_individuals:
            fp = _fingerprint(ind, tunable_names)
            ev = eval_cache[fp]
            population.append((ind, ev.fitness_score))

        best_fitness = max(f for _, f in population)
        worst_fitness = min(f for _, f in population)
        avg_fitness = sum(f for _, f in population) / len(population)
        logger.info(
            f"[GeneticStage] Init: {len(population)} individuals"
            f"  best={best_fitness:.4f}  worst={worst_fitness:.4f}  avg={avg_fitness:.4f}"
        )

        # best_history tracks best fitness: [init_best, after_gen0, after_gen1, ...]
        best_history = [best_fitness]

        # --- 3. Incremental loop ---
        actual_gens = 0
        for gen in range(self.n_generations):
            # 3a. Generate batch_size children
            children = []
            for _ in range(self.batch_size):
                parent_a = self._tournament_select(population, rng)
                parent_b = self._tournament_select(population, rng)
                child = self._crossover(parent_a, parent_b, space, rng)
                child = self._mutate(child, space, rng)
                children.append(child)

            # 3b. Dedup children against eval_cache
            novel_children = []
            novel_indices = []
            for i, child in enumerate(children):
                fp = _fingerprint(child, tunable_names)
                if fp not in eval_cache:
                    novel_children.append(child)
                    novel_indices.append(i)

            # 3c. Evaluate novel children
            if novel_children:
                evals = batch_runner(novel_children)
                for j, ev in enumerate(evals):
                    ev.stage = "genetic"
                    ev.stage_detail = f"gen_{gen}"
                    score_evaluation(ev, fitness_fn)
                    fp = _fingerprint(novel_children[j], tunable_names)
                    eval_cache[fp] = ev
                    all_evals.append(ev)

            # 3d. Worst replacement for ALL children (novel or cached)
            accepted = 0
            for child in children:
                fp = _fingerprint(child, tunable_names)
                child_ev = eval_cache[fp]
                child_fitness = child_ev.fitness_score

                # Find worst in population
                worst_idx = min(range(len(population)), key=lambda i: population[i][1])
                if child_fitness > population[worst_idx][1]:
                    population[worst_idx] = (child, child_fitness)
                    accepted += 1

            # 3e. Log
            cur_best = max(f for _, f in population)
            cur_worst = min(f for _, f in population)
            cur_avg = sum(f for _, f in population) / len(population)
            logger.info(
                f"[GeneticStage] Gen {gen}"
                f"  best={cur_best:.4f}  worst={cur_worst:.4f}  avg={cur_avg:.4f}"
                f"  accepted={accepted}/{self.batch_size}"
            )

            # 3f. Track best history
            best_history.append(cur_best)

            # 3g. Early stopping check
            actual_gens = gen + 1
            if self._check_early_stopping(best_history, gen):
                break

        # --- 4. Final log ---
        final_best = max(f for _, f in population)
        logger.info(
            f"[GeneticStage] Complete. {len(all_evals)} evaluations"
            f" ({actual_gens} generations). best={final_best:.4f}"
        )
        return all_evals

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_population(
        self, space: ParameterSpace, previous_results, rng, tunable_names,
    ) -> list[dict]:
        population = []

        if self.seed_from_previous and previous_results:
            valid = [
                ev for ev in previous_results
                if ev.fitness_score is not None and ev.fitness_score != float("-inf")
            ]
            top = sorted(valid, key=lambda e: e.fitness_score, reverse=True)[
                : self.n_seeds
            ]
            for ev in top:
                # Extract only tunable params
                ind = {k: v for k, v in ev.parameters.items() if k in tunable_names}
                population.append(ind)
            logger.info(
                f"[GeneticStage] Seeded {len(population)} individuals from previous results"
            )

        # Fill remaining slots randomly
        while len(population) < self.population_size:
            population.append(_random_individual(space, rng))

        return population

    # ------------------------------------------------------------------
    # Selection / crossover / mutation
    # ------------------------------------------------------------------

    def _tournament_select(
        self, scored: list[tuple[dict, float]], rng: random.Random,
    ) -> dict:
        """Select one individual via tournament selection."""
        contestants = rng.sample(scored, min(self.tournament_size, len(scored)))
        winner = max(contestants, key=lambda x: x[1])
        return winner[0]

    def _crossover(
        self, parent_a: dict, parent_b: dict,
        space: ParameterSpace, rng: random.Random,
    ) -> dict:
        """Uniform crossover: each param independently from parent A or B."""
        if rng.random() > self.crossover_prob:
            return dict(parent_a)
        child = {}
        for p in space.parameters:
            child[p.name] = rng.choice([parent_a[p.name], parent_b[p.name]])
        return child

    def _mutate(
        self, individual: dict, space: ParameterSpace, rng: random.Random,
    ) -> dict:
        """Per-parameter mutation: replace with random valid grid value."""
        result = dict(individual)
        for p in space.parameters:
            if rng.random() < self.mutation_prob:
                result[p.name] = rng.choice(p.valid_values())
        return result

    # ------------------------------------------------------------------
    # Early stopping
    # ------------------------------------------------------------------

    def _check_early_stopping(self, best_history: list[float], gen: int) -> bool:
        if self.early_stopping_generations <= 0:
            return False
        if len(best_history) <= self.early_stopping_generations:
            return False
        current = best_history[-1]
        old = best_history[-(self.early_stopping_generations + 1)]
        if current - old < self.early_stopping_min_delta:
            logger.info(
                f"[GeneticStage] Early stopping: no improvement > {self.early_stopping_min_delta}"
                f" in last {self.early_stopping_generations} generations."
                f" Best={current:.4f} at gen_{gen}"
            )
            return True
        return False
