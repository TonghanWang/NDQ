from functools import partial
from smac.env import MultiAgentEnv
from smac_plus import StarCraft2Env, Tracker1Env, Join1Env
import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
	return env(**kwargs)


REGISTRY = {
	"sc2": partial(env_fn, env=StarCraft2Env),
	"tracker1": partial(env_fn, env=Tracker1Env),
	"join1": partial(env_fn, env=Join1Env),
}

if sys.platform == "linux":
	os.environ.setdefault("SC2PATH",
	                      os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
