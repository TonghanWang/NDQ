from modules.agents import REGISTRY as agent_REGISTRY
from modules.comm import REGISTRY as comm_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import math
import torch as th
import torch.distributions as D
import torch.nn.functional as F
import numpy as np
import random


# This multi-agent controller shares parameters between agents
class TarCommMAC:
	def __init__(self, scheme, groups, args):
		self.n_agents = args.n_agents
		self.args = args
		self.comm_embed_dim = args.comm_embed_dim
		input_shape_for_comm = 0
		if args.comm:
			input_shape, input_shape_for_comm = self._get_input_shape(scheme)
		else:
			input_shape = self._get_input_shape(scheme)

		self._build_agents(input_shape)
		self.agent_output_type = args.agent_output_type

		self.action_selector = action_REGISTRY[args.action_selector](args)

		self.hidden_states = None

		if args.comm:
			self.comm = comm_REGISTRY[self.args.comm_method](input_shape_for_comm, args)

		self.is_print_once = False

		if self.args.env_args['print_rew']:
			self.c_step = 0
			self.cut_mean_num_list = []

	def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, thres=0., prob=0.):
		# Only select actions for the selected batch elements in bs
		avail_actions = ep_batch["avail_actions"][:, t_ep]
		# if self.args.comm:
		#     agent_outputs, (_, _), _, _ = self.forward(ep_batch, t_ep, test_mode=test_mode)
		# else:
		agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode, thres=thres, prob=prob)
		chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env,
		                                                    test_mode=test_mode)
		return chosen_actions

	def forward(self, ep_batch, t, test_mode=False, thres=0., prob=0.):
		agent_inputs = self._build_inputs(ep_batch, t)

		if self.args.comm:
			(_, _), messages, _ = self._communicate(ep_batch.batch_size, agent_inputs,
			                                        self.hidden_states.detach(), test_mode,
			                                        thres=thres, prob=prob)
			agent_inputs = th.cat([agent_inputs, messages], dim=1)

		avail_actions = ep_batch["avail_actions"][:, t]
		agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

		# Softmax the agent outputs if they're policy logits
		# is not used in qmix
		if self.agent_output_type == "pi_logits":

			if getattr(self.args, "mask_before_softmax", True):
				# Make the logits for unavailable actions very negative to minimise their affect on the softmax
				reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
				agent_outs[reshaped_avail_actions == 0] = -1e10

			agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
			if not test_mode:
				# Epsilon floor
				epsilon_action_num = agent_outs.size(-1)
				if getattr(self.args, "mask_before_softmax", True):
					# With probability epsilon, we will pick an available action uniformly
					epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()

				agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
				              + th.ones_like(agent_outs) * self.action_selector.epsilon / epsilon_action_num)

				if getattr(self.args, "mask_before_softmax", True):
					# Zero out the unavailable actions
					agent_outs[reshaped_avail_actions == 0] = 0.0

		# shape = (bs, self.n_agents, -1)
		return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

	def init_hidden(self, batch_size):
		self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

	def parameters(self):
		return list(self.agent.parameters()) + list(self.comm.parameters())

	def load_state(self, other_mac):
		self.agent.load_state_dict(other_mac.agent.state_dict())
		self.comm.load_state_dict(other_mac.comm.state_dict())

	def cuda(self):
		self.agent.cuda()
		self.comm.cuda()

	def save_models(self, path):
		th.save(self.agent.state_dict(), "{}/agent.th".format(path))
		th.save(self.comm.state_dict(), "{}/comm.th".format(path))

	def load_models(self, path):
		self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
		self.comm.load_state_dict(th.load("{}/comm.th".format(path), map_location=lambda storage, loc: storage))

	def _build_agents(self, input_shape):
		self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

	def _build_inputs(self, batch, t):
		# Assumes homogenous agents with flat observations.
		# Other MACs might want to e.g. delegate building inputs to each agent
		# shape = (bs * self.n_agents, -1)
		bs = batch.batch_size
		inputs = []
		inputs.append(batch["obs"][:, t])  # b1av
		if self.args.obs_last_action:
			if t == 0:
				inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
			else:
				inputs.append(batch["actions_onehot"][:, t - 1])
		if self.args.obs_agent_id:
			inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

		inputs = th.cat([x.reshape(bs * self.n_agents, -1) for x in inputs], dim=1)
		return inputs

	def _get_input_shape(self, scheme):
		# shape = (bs * self.n_agents, -1 + self.comm_embed_dim * self.n_agents)
		input_shape = scheme["obs"]["vshape"]
		ms_shape = 0
		if self.args.obs_last_action:
			input_shape += scheme["actions_onehot"]["vshape"][0]
		if self.args.obs_agent_id:
			input_shape += self.n_agents
		if self.args.comm:
			ms_shape = self.comm_embed_dim

		if self.args.comm:
			return input_shape + ms_shape, input_shape
		else:
			return input_shape

	def _cut_alpha(self, attn, thres=0., prob=0.):
		if self.args.is_cur_mu:
			s_num = 0
			attn = attn.detach().cpu()
			attn_2 = attn.detach().numpy()
			for i in range(self.n_agents):
				for j in range(self.n_agents):
					if i != j:
						if attn_2[0, i, j] <= thres or random.random() < prob:
							attn[0, i, j] = 0.
							s_num += 1
			attn = attn.cuda()
			self.c_step += 1
			self.cut_mean_num_list.append(1. * s_num / (self.n_agents * (self.n_agents - 1) + 1e-3))
			if self.c_step >= self.args.env_args['print_steps'] and not self.is_print_once:
				print('cut num ratio:', np.mean(self.cut_mean_num_list))
				self.c_step = 0
				self.is_print_once = True
		elif thres > 0:
			attn = attn.detach().cpu()
			index = np.argsort(abs(attn).reshape(-1))
			rank = np.zeros_like(index)
			for i, item in enumerate(index):
				rank[item] = i
			rank = 100. * rank / rank.size
			rank = th.Tensor(rank.reshape(attn.shape))
			attn[rank < thres] = 0.
			attn = attn.cuda()
		return attn

	def _communicate(self, bs, inputs, hidden_states, test_mode, thres=0., prob=0.):
		# shape = (bs * self.n_agents, -1)
		inputs = th.cat([inputs, hidden_states.view(bs * self.n_agents, -1)], dim=1).detach()
		message, signature, query = self.comm(inputs)

		message = message.view(bs, self.n_agents, -1)
		signature = signature.view(bs, self.n_agents, -1)
		query = query.view(bs, self.n_agents, -1)

		# Concatenate and weighting message

		if test_mode:
			all_m = message.clone()
			scores = th.matmul(query, signature.transpose(-2, -1)) / math.sqrt(self.args.comm_embed_dim)
			attn = F.softmax(scores, dim=-1)
			attn = self._cut_alpha(attn, thres=thres, prob=prob)
			comm = th.matmul(attn, all_m)
			comm = comm.reshape(bs * self.n_agents, -1)
			return (None, None), comm, None
		else:
			all_m = message.clone()
			scores = th.matmul(query, signature.transpose(-2, -1)) / math.sqrt(self.args.comm_embed_dim)
			attn = F.softmax(scores, dim=-1)
			comm = th.matmul(attn, all_m)
			comm = comm.reshape(bs * self.n_agents, -1)
			return (None, None), comm, None

	def clean(self):
		if self.args.env_args['print_rew']:
			self.c_step = 0
			self.cut_mean_num_list = []
			self.is_print_once = False
