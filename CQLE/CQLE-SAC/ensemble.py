from agent import CQLSAC
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from networks import Critic, Actor
import numpy as np
import math
import copy

class CQLEnsemble():

    def __init__(
            self,
            state_size,
            action_size,
            tau,
            hidden_size,
            learning_rate,
            temp,
            with_lagrange,
            cql_weight,
            target_action_gap,
            device,
            num_agents=5,
            is_GMM=False,
            s=1,
    ):
        self.device = device
        self.num_agents = num_agents
        self.is_GMM = is_GMM
        self.s = s
        self.CQL_agents = []
        for i in range(self.num_agents):
            CQL_agent = CQLSAC(
                state_size,
                action_size,
                tau,
                hidden_size,
                learning_rate,
                temp,
                with_lagrange,
                cql_weight,
                target_action_gap,
                device
            )
            self.CQL_agents.append(CQL_agent)

    def get_action(self, state, eval=False):

        state = torch.from_numpy(state).float().to(self.device)

        # TODO: implement the function to calculate each agent's confidence of the state
        confidences = np.arange(1, self.num_agents)

        with torch.no_grad():
            if eval:
                actions = [self.CQL_agents[i].actor_local.get_det_action(state).numpy() for i in range(self.num_agents)]
            else:
                actions = [self.CQL_agents[i].actor_local.get_action(state).numpy() for i in range(self.num_agents)]

        return self.vote(actions, confidences)

    def vote(self, actions, confidences):
        # TODO: implement different voting strategies to get the action
        return actions[0]

    def learn(self, experiences):
        for i in range(self.num_agents):
            self.num_agents[i].learn(experiences[i])
