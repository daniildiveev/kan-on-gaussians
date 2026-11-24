"""Individual attempt results and fold-weighted summaries."""

from dataclasses import dataclass
from statistics import fmean

from src.experiments.series import Trial


@dataclass(frozen=True)
class TrialResult:
    trial: Trial
    run_id: str
    best_epoch: int
    test_rmse: float | None
    test_r2: float | None
    best_val_rmse: float | None = None


@dataclass(frozen=True)
class SeriesSummary:
    series_id: str
    results: tuple[TrialResult, ...]
    pending: tuple[Trial, ...]
    evaluation: str = "test"

    @property
    def by_model(self) -> dict[str, dict]:
        """Average repeats within each fold, then weight observed folds equally."""
        planned = [r.trial for r in self.results] + list(self.pending)
        summary = {}
        for model in dict.fromkeys(trial.model for trial in planned):
            runs = [r for r in self.results if r.trial.model == model]
            metrics = ("best_val_rmse",) if self.evaluation == "validation" else ("test_rmse", "test_r2")
            fold_means = {metric: [] for metric in metrics}
            for fold in sorted({r.trial.fold for r in runs}):
                fold_runs = [r for r in runs if r.trial.fold == fold]
                for metric, means in fold_means.items():
                    values = [getattr(r, metric) for r in fold_runs]
                    defined = [value for value in values if value is not None]
                    if defined:
                        means.append(fmean(defined))
            summary[model] = {
                "completed": len(runs),
                "planned": sum(t.model == model for t in planned),
                **{name: fmean(means) if means else None for name, means in fold_means.items()},
            }
        return summary

