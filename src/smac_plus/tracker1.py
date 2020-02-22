from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env.multiagentenv import MultiAgentEnv

import atexit
from operator import attrgetter
from copy import deepcopy
import numpy as np
import enum
import math
from absl import logging
import random

actions = {
	"move": 16,  # target: PointOrUnit
	"attack": 23,  # target: PointOrUnit
	"stop": 4,  # target: None
	"heal": 386,  # Unit
}


class Direction(enum.IntEnum):
	NORTH = 0
	SOUTH = 1
	EAST = 2
	WEST = 3


class Tracker1Env(MultiAgentEnv):
	"""The StarCraft II environment for decentralised multi-agent
	micromanagement scenarios.
	"""

	def __init__(
			self,
			difficulty="7",
			seed=None,
			obs_last_action=False,
			obs_pathing_grid=False,
			obs_terrain_height=False,
			obs_instead_of_state=False,
			state_last_action=True,
			reward_sparse=False,
			reward_only_positive=True,
			reward_death_value=10,
			reward_win=200,
			reward_defeat=0,
			reward_negative_scale=0.5,
			reward_scale=True,
			reward_scale_rate=20,
			replay_dir="",
			replay_prefix="",
			window_size_x=1920,
			window_size_y=1200,
			partial_obs=True,
			is_print=False,
			communication=False,
			debug=False,
			print_rew=False,
			print_steps=1000
	):
		# Map arguments
		self.n_agents = 3

		# Observations and state
		self.obs_instead_of_state = obs_instead_of_state
		self.obs_last_action = obs_last_action
		self.obs_pathing_grid = obs_pathing_grid
		self.obs_terrain_height = obs_terrain_height
		self.state_last_action = state_last_action

		# Rewards args
		self.reward_sparse = reward_sparse
		self.reward_only_positive = reward_only_positive
		self.reward_negative_scale = reward_negative_scale
		self.reward_death_value = reward_death_value
		self.reward_win = reward_win
		self.reward_defeat = reward_defeat
		self.reward_scale = reward_scale
		self.reward_scale_rate = reward_scale_rate

		# Other
		self._seed = random.randint(0, 9999)
		np.random.seed(self._seed)
		self.debug = debug
		self.window_size = (window_size_x, window_size_y)
		self.replay_dir = replay_dir
		self.replay_prefix = replay_prefix

		# Actions
		self.n_actions = 5
		self.is_print = is_print
		self.print_rew = print_rew
		self.print_steps = print_steps

		# Map info
		# self._agent_race = map_params["a_race"]
		# self._bot_race = map_params["b_race"]
		# self.map_type = map_params["map_type"]

		self._episode_count = 0
		self._episode_steps = 0
		self._total_steps = 0
		self.battles_won = 0
		self.battles_game = 0
		self.last_stats = None
		self.previous_ally_units = None
		self.previous_enemy_units = None
		self.last_action = np.zeros((self.n_agents, self.n_actions))
		self._min_unit_type = 0
		self.max_distance_x = 0
		self.max_distance_y = 0

		self.target2 = np.random.randint(2)
		self.battles_game = 0
		self.battles_won = 0
		self.partial_obs = partial_obs
		self.communication = communication
		self.episode_limit = 1

		self.p_step = 0
		self.rew_gather = []
		self.is_print_once = False

		# Try to avoid leaking SC2 processes on shutdown
		# atexit.register(lambda: self.close())

	def step(self, actions):
		"""Returns reward, terminated, info."""

		if self.is_print:
			print('target: ', self.target2)
			print('*********************')

		reward = 0

		if self.target2:
			if actions[1] == 3 and actions[2] == 1:
				reward += 30
				self.battles_won += 1
			elif actions[1] == 1 and actions[0] == 3:
				reward += 20
		else:
			if actions[1] == 1 and actions[0] == 3:
				self.battles_won += 1
				reward += 20

		for action in actions:
			if action <= 3:
				reward -= 5

		if self.print_rew:
			self.p_step += 1
			self.rew_gather.append(reward)
			if self.p_step % self.print_steps == 0:
				print('steps: %d, average rew: %.3lf' % (self.p_step,
				                                         float(np.mean(self.rew_gather))))
				self.is_print_once = True

		return reward, True, {'battle_won': int(reward > 0)}

	def get_obs(self):
		"""Returns all agent observations in a list."""
		if self.partial_obs:
			return [self.get_obs_agent(i) for i in range(self.n_agents)]
		else:
			return [self.get_f_agent() for i in range(self.n_agents)]

	# def get_obs_comm_agent(self, agent_id):
	#     """Returns observation for agent_id."""
	#     obs = []
	#     if agent_id == 0:
	#         obs = [-1, 1, self.target2]
	#     elif agent_id == 1:
	#         obs = [1, self.target2, 0]
	#     else:
	#         obs = [self.target2, -1, 0]
	#
	#     return np.array(obs)

	def get_obs_agent(self, agent_id):
		"""Returns observation for agent_id."""
		obs = []
		if agent_id == 0:
			obs = [-1, 1]
		elif agent_id == 1:
			obs = [1, self.target2]
		else:
			obs = [self.target2, -1]

		return np.array(obs)

	# def get_o3_agent(self, agent_id):
	#     """Returns observation for agent_id."""
	#     obs = []
	#     if agent_id == 0:
	#         obs = [-1, 1, 0]
	#     elif agent_id == 1:
	#         obs = [1, self.target2, 0]
	#     else:
	#         obs = [self.target2, -1, 0]
	#
	#     return np.array(obs)

	# def get_f_agent(self, agent_id):
	#     return np.array([1, self.target2, -1])

	def get_obs_size(self):
		"""Returns the size of the observation."""

		# wth: the very first trial of facomm project, comm, f, o, o3, and f3
		# if self.communication:
		#     return 3
		# elif not self.partial_obs:
		#     return 3
		# else:
		#     return 3

		# wth: normally, it is 2
		return 2

	def get_state(self):
		"""Returns the global state."""
		return np.array([1, self.target2])

	def get_state_size(self):
		"""Returns the size of the global state."""
		return 2

	def get_avail_actions(self):
		"""Returns the available actions of all agents in a list."""
		return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

	def get_avail_agent_actions(self, agent_id):
		"""Returns the available actions for agent_id."""
		return [1] * self.n_actions

	def get_total_actions(self):
		"""Returns the total number of actions an agent could ever take."""
		return self.n_actions

	def reset(self):
		"""Returns initial observations and states."""
		self.target2 = np.random.randint(2)
		self.battles_game += 1

		return self.get_obs(), self.get_state()

	def render(self):
		pass

	def close(self):
		pass

	def seed(self):
		pass

	def save_replay(self):
		"""Save a replay."""
		pass

	def get_env_info(self):
		env_info = {"state_shape": self.get_state_size(),
		            "obs_shape": self.get_obs_size(),
		            "n_actions": self.get_total_actions(),
		            "n_agents": self.n_agents,
		            "episode_limit": 2}
		return env_info

	def get_stats(self):
		stats = {
			"battles_won": self.battles_won,
			"battles_game": self.battles_game,
			"win_rate": self.battles_won / self.battles_game
		}
		return stats

	def clean(self):
		self.p_step = 0
		self.rew_gather = []
		self.is_print_once = False
