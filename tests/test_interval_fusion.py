from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from interval_fusion import IntervalFusionProblem, optimize_interval_fusion, interval_combine
from metrics import interval_metrics, high_flow_metrics


def load_series(name):
    return np.loadtxt(
        ROOT / "test_data" / "interval_fusion" / name,
        delimiter=",",
        skiprows=1,
        usecols=1,
    )


obs = load_series("observed_streamflow.csv")
cl = load_series("conformal_lower.csv")
cu = load_series("conformal_upper.csv")
bl = load_series("bayesian_lower.csv")
bu = load_series("bayesian_upper.csv")

threshold = float(np.quantile(obs, 0.75))
population_size = 40

problem = IntervalFusionProblem(
    confidence=0.90,
    upper_1=cu,
    lower_1=cl,
    upper_2=bu,
    lower_2=bl,
    observed=obs,
    flood_threshold=threshold,
    population_size=population_size,
    strategy="improved",
)

# Small verification run. Increase max_evaluations for research experiments.
result = optimize_interval_fusion(
    problem,
    optimizer="NSGAII",
    population_size=population_size,
    max_evaluations=1600,
)

assert len(result) > 0

solution = result[0]
upper, lower = interval_combine(solution.variables, cu, cl, bu, bl)
picp, pinaw, _ = interval_metrics(upper, lower, obs)
fpicp, _, _, sis = high_flow_metrics(upper, lower, obs, threshold)

print("Number of nondominated solutions:", len(result))
print("PICP:", picp)
print("PINAW:", pinaw)
print("High-flow PICP:", fpicp)
print("SIS:", sis)
print("Interval-fusion test passed.")
