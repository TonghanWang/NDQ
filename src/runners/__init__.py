REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
REGISTRY["parallel"] = ParallelRunner

from .episode_runner_full import EpisodeRunner_full
REGISTRY["episode_full"] = EpisodeRunner_full

from .parallel_runner_x import ParallelRunner_x
REGISTRY["parallel_x"] = ParallelRunner_x
