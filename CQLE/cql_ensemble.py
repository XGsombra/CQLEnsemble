from rlkit.torch.sac.cql import CQLTrainer
from dataset_spliter import split_dataset

from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

import rlkit.torch.pytorch_util as ptu
from rlkit.core.eval_util import create_stats_ordered_dict
from rlkit.torch.torch_rl_algorithm import TorchTrainer
from torch import autograd


class CQLETrainer():
    def __init__(
            self,
            env,
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            discount=0.99,
            reward_scale=1.0,

            policy_lr=1e-3,
            qf_lr=1e-3,
            optimizer_class=optim.Adam,

            soft_target_tau=1e-2,
            plotter=None,
            render_eval_paths=False,

            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,
            num_qs=2,

            # CQL
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            # CQL Ensemble
            num_agents=10,
            is_GMM=True,
            scatterness=1,

            ## sort of backup
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0,
    ):
        self.num_agents = num_agents
        self.is_GMM = is_GMM
        self.scatterness = scatterness

        # Create the cql agents
        if self.num_agents < 1:
            raise Exception("Error: At least one CQL agent is required.")
        self.agents = []
        for i in range(self.num_agents):
            cql_agent = CQLTrainer(
                env,
                policy,
                qf1,
                qf2,
                target_qf1,
                target_qf2,

                discount=discount,
                reward_scale=reward_scale,

                policy_lr=policy_lr,
                qf_lr=qf_lr,
                optimizer_class=optimizer_class,

                soft_target_tau=soft_target_tau,
                plotter=plotter,
                render_eval_paths=render_eval_paths,

                use_automatic_entropy_tuning=use_automatic_entropy_tuning,
                target_entropy=target_entropy,
                policy_eval_start=policy_eval_start,
                num_qs=num_qs,

                min_q_version=min_q_version,
                temp=temp,
                min_q_weight=min_q_weight,

                ## sort of backup
                max_q_backup=max_q_backup,
                deterministic_backup=deterministic_backup,
                num_random=num_random,
                with_lagrange=with_lagrange,
                lagrange_thresh=lagrange_thresh,
            )
            self.agents.append(cql_agent)

    def train(self, dataset):
        observations = dataset['observations']
        next_obs = dataset['next_observations']
        actions = dataset['actions']
        rewards = np.expand_dims(np.squeeze(dataset['rewards']), 1)
        terminals = np.expand_dims(np.squeeze(dataset['terminals']), 1)
        N = observations.shape[0]
        X = observations
        y = []
        for i in range(N):
            y.append([next_obs[i], actions[i], rewards[i], terminals[i]])
        wrapped_datasets = split_dataset(
            X,
            y,
            num_datasets=self.num_agents,
            is_GMM=self.is_GMM,
            scatterness=self.scatterness
        )

        # batches is a dictionary {agent_id: batch for the agent}
        for agent_id in range(self.num_agents):
            dataset['observations'] = wrapped_datasets[agent_id][0]
            dataset['next_observations'] = wrapped_datasets[agent_id][1][0]
            dataset['actions'] = wrapped_datasets[agent_id][1][1]
            dataset['rewards'] = wrapped_datasets[agent_id][1][2]
            dataset['terminals'] = wrapped_datasets[agent_id][1][3]
            self.agents[agent_id].train(dataset)
