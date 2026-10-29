"""Phase 8 Adaptive Perturbation Twin.

The package is intentionally separate from the Phase 6 causal operator and the
Phase 7 active-probing benchmark.  Phase 8 changes the experimental unit from a
target to an animal-conditional support set, and the endpoint from a scalar to a
post-stimulus trajectory.
"""

from neurobranch.twin.model import ResponseTwin, TwinConfig, TwinPrediction
from neurobranch.twin.tasks import AnimalTask, AnimalTaskSplit, build_animal_tasks
from neurobranch.twin.train import TwinFit, fit_twin
from neurobranch.twin.windows import (
    EventWindowCache,
    WindowConfig,
    build_event_windows,
    load_window_cache,
)

__all__ = [
    "AnimalTask",
    "AnimalTaskSplit",
    "EventWindowCache",
    "ResponseTwin",
    "TwinConfig",
    "TwinFit",
    "TwinPrediction",
    "WindowConfig",
    "build_animal_tasks",
    "build_event_windows",
    "fit_twin",
    "load_window_cache",
]
