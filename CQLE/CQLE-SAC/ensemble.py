from torch import Tensor

from agent import CQLSAC
import torch
import numpy as np


def vote(actions, confidences, qs, strategy):

    if strategy == "autocratic":
        # The agent with the max confidence will make the decision
        return actions[np.argmax(confidences)]

    q_mean_across_agents = np.mean(qs, axis=1)

    if strategy == "aristocratic":
        # Only the agents with over-average confidence could vote
        candidate_indices = q_mean_across_agents > np.mean(q_mean_across_agents)
        candidate_qs = q_mean_across_agents[candidate_indices]
        candidate_actions = actions[candidate_indices]
        return candidate_actions[np.argmax(candidate_qs)]

    if strategy == "meritocratic":
        # uses the weighed sum of the actions proposed by each agent
        weights = q_mean_across_agents * confidences
        weights = weights / np.sum(weights)
        return np.sum(actions * weights[:, np.newaxis], axis=0)

    return actions[0]


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
            strategy="autocratic",
            pca=None,
            action_sample_num=1000
    ):
        # Ensemble Attributes
        self.device = device
        self.num_agents = num_agents
        self.is_GMM = is_GMM
        self.s = s
        self.strategy = strategy
        self.pca = pca
        self.CQL_agents = []
        self.means = []
        self.covariances = []
        self.action_sample_num = action_sample_num

        # CQL Agents
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
        with torch.no_grad():
            if eval:
                actions = Tensor(np.vstack([self.CQL_agents[i].actor_local.get_det_action(state).numpy() for i in range(self.num_agents)]))
            else:
                actions = Tensor(np.vstack([self.CQL_agents[i].actor_local.get_action(state).numpy() for i in range(self.num_agents)]))

        actions = actions.to(self.device)

        # get the q-value for each action according each agent (double q trick applied)
        q1s = Tensor([[self.CQL_agents[i].critic1(state, action) for i in range(self.num_agents)] for action in actions])
        q1s = np.array([[self.CQL_agents[i].critic1(state, action).cpu().detach().numpy() for i in range(self.num_agents)] for action in actions])
        q2s = np.array([[self.CQL_agents[i].critic2(state, action).cpu().detach().numpy() for i in range(self.num_agents)] for action in actions])
        q1s = q1s.reshape((self.num_agents, self.num_agents))
        q2s = q2s.reshape((self.num_agents, self.num_agents))
        qs_min = np.amin([q1s, q2s], axis=0)

        # standardize the q-value distributions of each agent according to the mean and std of sample q-values
        action_dim = actions.size()[1]
        action_samples = (torch.rand((self.action_sample_num, action_dim)) * 2 - 1).to(self.device)
        q1s_samples = np.array([[self.CQL_agents[i].critic1(state, action_sample).cpu().detach().numpy() for i in range(self.num_agents)] for action_sample in action_samples])
        q2s_samples = np.array([[self.CQL_agents[i].critic2(state, action_sample).cpu().detach().numpy() for i in range(self.num_agents)] for action_sample in action_samples])
        qs_samples = np.amin([q1s_samples, q2s_samples], axis=0)
        qs_means = np.mean(qs_samples, axis=0)
        qs_stds = np.std(qs_samples, axis=0)
        qs_min_standardized = (qs_min - qs_means) / qs_stds


        # convert actions to numpy array in cpu
        actions = actions.cpu().detach().numpy()
        state = state.cpu().detach().numpy()

        # calculate the confidence according to the CDF of Gaussian distribution
        d = self.pca.n_components_
        pca_state = self.pca.transform(state[np.newaxis, :])
        dist = pca_state[np.newaxis, ...] - self.means[:, np.newaxis, :]
        distT = np.transpose(dist, axes=[0, 2, 1])
        log_numerator = (-dist @ np.linalg.inv(self.covariances) @ distT / 2).reshape((self.num_agents,))
        log_denominator = (np.log(np.linalg.det(self.covariances)) + d * np.log(2 * np.pi)) / 2
        confidences = np.exp(log_numerator - log_denominator)
        confidences = confidences / np.sum(confidences)

        return vote(actions, confidences, qs_min_standardized, self.strategy)

    def learn(self, experiences):
        for i in range(self.num_agents):
            self.num_agents[i].learn(experiences[i])
