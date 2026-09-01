"""
Uncertainty interval fusion and multi-objective optimization.

The released implementation accepts two externally generated prediction
intervals and does not include restricted observational data or the complete
Grid-Xinanjiang model.
"""

import numpy as np
from platypus import NSGAII, IBEA, SPEA2, GDE3, Problem, Real

try:
    from .metrics import interval_metrics, high_flow_metrics
except ImportError:
    from metrics import interval_metrics, high_flow_metrics


def interval_combine(params, upper_1, lower_1, upper_2, lower_2):
    """Second-order response-function fusion for upper and lower bounds."""
    a = np.asarray(params, dtype=float)

    upper = (
        a[0]
        + a[1] * upper_1
        + a[2] * upper_2
        + a[3] * upper_1**2
        + a[4] * upper_2**2
        + a[5] * upper_1 * upper_2
    )

    lower = (
        a[6]
        + a[7] * lower_1
        + a[8] * lower_2
        + a[9] * lower_1**2
        + a[10] * lower_2**2
        + a[11] * lower_1 * lower_2
    )
    return upper, lower


def optimization_weight(start_generation, end_generation, generation, lower=0.1, upper=1.0):
    """Generation-dependent linear weight constrained to [lower, upper]."""
    if generation < start_generation:
        return lower
    if generation >= end_generation:
        return upper
    if end_generation <= start_generation:
        return upper
    fraction = (generation - start_generation) / (end_generation - start_generation)
    return lower + (upper - lower) * fraction


class IntervalFusionProblem(Problem):
    """
    Three-objective interval-fusion problem.

    Objective 1: deviation of overall PICP from the target coverage.
    Objective 2: PINAW.
    Objective 3: SIS under high-flow conditions.

    Under the improved strategy, the three objectives receive
    generation-dependent weights. Under the basic strategy, all weights equal 1.
    """

    def __init__(
        self,
        confidence,
        upper_1,
        lower_1,
        upper_2,
        lower_2,
        observed,
        flood_threshold,
        population_size=200,
        strategy="improved",
        coverage_offset=0.03,
        penalty_factor=100.0,
    ):
        super().__init__(12, 3)
        self.types[:] = [Real(-1.0, 1.0) for _ in range(12)]
        self.directions[:] = [Problem.MINIMIZE] * 3

        self.confidence = float(confidence)
        self.upper_1 = np.asarray(upper_1, dtype=float)
        self.lower_1 = np.asarray(lower_1, dtype=float)
        self.upper_2 = np.asarray(upper_2, dtype=float)
        self.lower_2 = np.asarray(lower_2, dtype=float)
        self.observed = np.asarray(observed, dtype=float)
        self.flood_threshold = float(flood_threshold)

        self.population_size = int(population_size)
        self.strategy = strategy.lower()
        self.coverage_offset = float(coverage_offset)
        self.penalty_factor = float(penalty_factor)
        self.evaluation_count = 0

    def _generation(self):
        return self.evaluation_count // max(self.population_size, 1)

    def evaluate(self, solution):
        self.evaluation_count += 1
        generation = self._generation()

        upper, lower = interval_combine(
            solution.variables,
            self.upper_1,
            self.lower_1,
            self.upper_2,
            self.lower_2,
        )

        violation = np.sum(np.maximum(0.0, lower - upper))
        penalty = violation * self.penalty_factor

        picp, pinaw, _ = interval_metrics(upper, lower, self.observed)
        _, _, _, sis = high_flow_metrics(
            upper,
            lower,
            self.observed,
            self.flood_threshold,
        )

        if np.isnan(sis):
            sis = 10.0

        if self.strategy == "improved":
            w_picp = optimization_weight(0, 20, generation)
            w_sis = optimization_weight(10, 30, generation)
            w_pinaw = optimization_weight(20, 30, generation)
        elif self.strategy == "basic":
            w_picp = w_sis = w_pinaw = 1.0
        else:
            raise ValueError("strategy must be 'basic' or 'improved'")

        target_coverage = self.confidence + self.coverage_offset

        solution.objectives[0] = w_picp * abs(picp - target_coverage) + penalty
        solution.objectives[1] = w_pinaw * pinaw + penalty
        solution.objectives[2] = w_sis * sis + penalty


def optimize_interval_fusion(
    problem,
    optimizer="NSGAII",
    population_size=200,
    max_evaluations=10000,
):
    """
    Run the selected multi-objective optimizer.

    NSGA-II is the primary optimizer used by the proposed framework.
    Other optimizers are exposed for comparison experiments.
    """
    algorithms = {
        "NSGAII": NSGAII,
        "IBEA": IBEA,
        "SPEA2": SPEA2,
        "GDE3": GDE3,
    }

    if optimizer not in algorithms:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    algorithm = algorithms[optimizer](problem, population_size=population_size)
    algorithm.run(max_evaluations)
    return algorithm.result
