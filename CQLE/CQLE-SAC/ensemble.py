from torch import Tensor

from agent import CQLSAC
import torch
import numpy as np

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
        self.means = []
        self.covariances = []
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

        # get the q-values for each action according each agent
        q1s = np.array([[self.CQL_agents[i].critic1(state, action).cpu().detach().numpy() for i in range(self.num_agents)] for action in actions])
        q2s = np.array([[self.CQL_agents[i].critic2(state, action).cpu().detach().numpy() for i in range(self.num_agents)] for action in actions])

        # convert actions to numpy array in cpu
        actions = actions.cpu().detach().numpy()
        state = state.cpu().detach().numpy()

        # calculate the confidence
        d = state.shape[0]
        dist = state[np.newaxis, np.newaxis, :] - self.means[:, np.newaxis, :]
        distT = np.transpose(dist, axes=[0, 2, 1])
        log_numerator = (-dist @ np.linalg.inv(self.covariances) @ distT / 2).reshape((self.num_agents,))
        log_denominator = (np.linalg.det(self.covariances) + d * np.log(2 * np.pi)) / 2
        log_confidences = log_numerator - log_denominator
        print(log_confidences)

        return self.vote(actions, log_confidences, q1s, q2s)

    def vote(self, actions, confidences, q1s, q2s):
        # TODO: implement different voting strategies to get the action
        return actions[0]

    def learn(self, experiences):
        for i in range(self.num_agents):
            self.num_agents[i].learn(experiences[i])
