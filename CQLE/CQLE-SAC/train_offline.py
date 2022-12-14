import gym
import d4rl
import numpy as np
from collections import deque
import torch
from torch import Tensor
import time

import wandb
import argparse
import glob
import random
from torch.utils.data import DataLoader, TensorDataset
from dataset_splitter import split_dataset
from ensemble import CQLEnsemble


def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL", help="Run name, default: CQL")
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2",
                        help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0,
                        help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    parser.add_argument("--num_agents", type=int, default=5, help="")
    parser.add_argument("--is_GMM", type=int, default=0, help="")
    parser.add_argument("--s", type=float, default=1.0, help="")
    parser.add_argument("--strategy", type=str, default="autocratic", help="The strategy to vote")
    parser.add_argument("--pca_n", type=int, default=2, help="The pca component number")
    parser.add_argument("--standardize_q", type=int, default=1, help="1 if standardize q-values across agents")

    args = parser.parse_args()
    return args


def prep_dataloaders(config):
    env = gym.make(config.env)
    eval_env = env
    eval_env.seed(config.seed)
    dataset = d4rl.qlearning_dataset(env.unwrapped)
    tensors = {}
    for k, v in dataset.items():
        if k in ["actions", "observations", "next_observations", "rewards", "terminals"]:
            if k is not "terminals":
                tensors[k] = torch.from_numpy(v).float()
            else:
                tensors[k] = torch.from_numpy(v).long()

    datasets, pca = split_dataset(
        tensors["observations"],
        tensors["actions"],
        tensors["rewards"][:, None],
        tensors["next_observations"],
        tensors["terminals"][:, None],
        num_datasets=config.num_agents,
        is_GMM=config.is_GMM,
        s=config.s,
        pca_n_component=config.pca_n
    )

    dataloaders = []
    for i in range(config.num_agents):
        print(f"Dataset {i} has {len(datasets[i]['observations'])} entries.")
        tensordata = TensorDataset(datasets[i]["observations"],
                                   datasets[i]["actions"],
                                   datasets[i]["rewards"],
                                   datasets[i]["next_observations"],
                                   datasets[i]["terminals"])
        dataloaders.append(DataLoader(tensordata, batch_size=config.batch_size, shuffle=True))

    print("------------------------------------dataloaders generated------------------------------")

    return dataloaders, eval_env, pca


def evaluate(env, policy, eval_runs=5, is_ensemble=False):
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):

        env.seed(i)
        state = env.reset()

        if is_ensemble:
            # standardize the q-value distributions of each agent according to the mean and std of sample q-values
            action_dim = env.action_space.shape[0]
            action_samples = (torch.rand((policy.action_sample_num, action_dim)) * 2 - 1).to(policy.device)
            q1s_samples = np.array([[policy.CQL_agents[i].critic1(policy.obs_means[i].to(policy.device), action_sample).cpu().detach().numpy() for i in range(policy.num_agents)] for action_sample in action_samples])
            q2s_samples = np.array([[policy.CQL_agents[i].critic2(policy.obs_means[i].to(policy.device), action_sample).cpu().detach().numpy() for i in range(policy.num_agents)] for action_sample in action_samples])
            qs_samples = np.amin([q1s_samples, q2s_samples], axis=0)
            policy.qs_sample_means = np.mean(qs_samples, axis=0)
            policy.qs_sample_stds = np.std(qs_samples, axis=0)

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)

            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)


def reformat_dataloaders(dataloaders):
    # [(batch_idx, [batched experiences for all agents])]
    num_agents = len(dataloaders)
    extracted_datasets = [[batch for _, batch in enumerate(dataloaders[i])] for i in range(num_agents)]
    max_batch_num = max([len(h) for h in extracted_datasets])
    new_dataset = []
    for i in range(max_batch_num):
        batch_i = []
        for agent_id in range(num_agents):
            batch_i.append(extracted_datasets[agent_id][i] if len(extracted_datasets[agent_id]) > i else [])
        new_dataset.append((i, batch_i))
    return new_dataset


def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataset_preparation_start_time = time.time()
    dataloaders, env, pca = prep_dataloaders(config)
    dataloaders_preparation_time = time.time() - dataset_preparation_start_time

    env.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"--------------------------device is {device}------------------------------")

    # Initialize some statistics
    batches = 0
    average10 = deque(maxlen=10)
    training_times = []
    evaluation_times = []

    run_name = f"{config.strategy}-{'gmm' if config.is_GMM else 'rnd'}-{config.num_agents}-agents-{config.s}-s-{config.episodes}-epochs-{'l' if config.with_lagrange else ''}-{'' if config.standardize_q else 'no-st'}"

    with wandb.init(project="CQL-ensemble-offline-halfcheetah", name=run_name, config=config):

        ensemble = CQLEnsemble(
            state_size=env.observation_space.shape[0],
            action_size=env.action_space.shape[0],
            tau=config.tau,
            hidden_size=config.hidden_size,
            learning_rate=config.learning_rate,
            temp=config.temperature,
            with_lagrange=config.with_lagrange,
            cql_weight=config.cql_weight,
            target_action_gap=config.target_action_gap,
            device=device,
            num_agents=config.num_agents,
            is_GMM=config.is_GMM == 1,
            s=config.s,
            strategy=config.strategy,
            pca=pca,
            standardize_q=config.standardize_q == 1
        )

        # Calculate the mean and covariance matrix of each agent
        print("--------------------------started to calculate means and variances------------------------------")
        for i in range(ensemble.num_agents):
            dataset_size = len(dataloaders[i].dataset)
            dataset_as_array = np.vstack([np.array(dataloaders[i].dataset[j][0]) for j in range(dataset_size)])
            ensemble.means.append(np.mean(ensemble.pca.transform(dataset_as_array), axis=0))
            ensemble.covariances.append(np.cov(ensemble.pca.transform(dataset_as_array).T))
            ensemble.obs_means.append(Tensor(np.mean(dataset_as_array, axis=0)))
        ensemble.means = np.array(ensemble.means)
        ensemble.covariances = np.array(ensemble.covariances)
        print("------------------------finished calculating means and variances------------------------------")

        # wandb.watch(ensemble, log="gradients", log_freq=10)
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x % 10 == 0, force=True)

        print("--------------------------started to evaluate the untrained model------------------------------")
        eval_reward = evaluate(env, ensemble, is_ensemble=True)
        wandb.log({"Test Reward": eval_reward, "Episode": 0, "Batches": batches}, step=batches)
        for agent_id in range(ensemble.num_agents):
            wandb.log({f"Reward for Agent {agent_id}": evaluate(env, ensemble.CQL_agents[agent_id])})
        print("--------------------------finished evaluating the untrained model------------------------------")

        for i in range(1, config.episodes + 1):

            dataset = reformat_dataloaders(dataloaders)

            for batch_idx, experiences in dataset:
                total_policy_loss = []
                total_alpha_loss = []
                total_bellmann_error1 = []
                total_bellmann_error2 = []
                total_cql1_loss = []
                total_cql2_loss = []
                total_current_alpha = []
                total_lagrange_alpha_loss = []
                total_lagrange_alpha = []

                num_training_agents = 0

                training_time_start = time.time()
                for agent_id in range(ensemble.num_agents):
                    experience = experiences[agent_id]
                    if len(experience) == 0:
                        continue
                    num_training_agents += 1
                    states, actions, rewards, next_states, dones = experience
                    states = states.to(device)
                    actions = actions.to(device)
                    rewards = rewards.to(device)
                    next_states = next_states.to(device)

                    dones = dones.to(device)
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = \
                    ensemble.CQL_agents[agent_id].learn((states, actions, rewards, next_states, dones))
                    batches += 1

                    total_policy_loss.append(policy_loss)
                    total_alpha_loss.append(alpha_loss)
                    total_bellmann_error1.append(bellmann_error1)
                    total_bellmann_error2.append(bellmann_error2)
                    total_cql1_loss.append(cql1_loss)
                    total_cql2_loss.append(cql2_loss)
                    total_current_alpha.append(current_alpha)
                    total_lagrange_alpha_loss.append(lagrange_alpha_loss)
                    total_lagrange_alpha.append(lagrange_alpha)

                training_times.append(time.time() - training_time_start)
                policy_loss = sum(total_policy_loss) / num_training_agents
                alpha_loss = sum(total_alpha_loss) / num_training_agents
                bellmann_error1 = sum(total_bellmann_error1) / num_training_agents
                bellmann_error2 = sum(total_bellmann_error2) / num_training_agents
                cql1_loss = sum(total_cql1_loss) / num_training_agents
                cql2_loss = sum(total_cql2_loss) / num_training_agents
                current_alpha = sum(total_current_alpha) / num_training_agents
                lagrange_alpha_loss = sum(total_lagrange_alpha_loss) / num_training_agents
                lagrange_alpha = sum(total_lagrange_alpha) / num_training_agents

            if i % config.eval_every == 0:
                print(f"Started evaluation for episode {i}")
                start_time = time.process_time()
                # If evaluating all voting strategies
                if ensemble.strategy == "all":
                    for standardize_q in [True, False]:
                        ensemble.standardize_q = standardize_q
                        for strategy in ["autocratic", "aristocratic", "meritocratic"]:
                            ensemble.strategy = strategy
                            reward = evaluate(env, ensemble, is_ensemble=True)
                            print(f"{strategy} strategy has reward of {reward}")
                            wandb.log({f"Reward of {strategy}{'with Q-standardization' if standardize_q else ''}": reward})
                    ensemble.strategy = "all"
                else:
                    wandb.log({"Test Reward": evaluate(env, ensemble, is_ensemble=True)})
                evaluation_times.append(time.process_time() - start_time)
                for agent_id in range(ensemble.num_agents):
                    wandb.log({f"Reward for Agent {agent_id}": evaluate(env, ensemble.CQL_agents[agent_id])})
                wandb.log({"Episode": i, "Batches": batches}, step=batches)

                average10.append(eval_reward)
                print("Episode: {} | Polciy Loss: {} | Batches: {}".format(i, policy_loss,
                                                                                        batches, ))

            wandb.log({
                "Average10": np.mean(average10),
                "Policy Loss": policy_loss,
                "Alpha Loss": alpha_loss,
                "Lagrange Alpha Loss": lagrange_alpha_loss,
                "CQL1 Loss": cql1_loss,
                "CQL2 Loss": cql2_loss,
                "Bellman error 1": bellmann_error1,
                "Bellman error 2": bellmann_error2,
                "Alpha": current_alpha,
                "Lagrange Alpha": lagrange_alpha,
                "Batches": batches,
                "Episode": i,
                "Training time": training_times[-1],
                "Evaluation time": evaluation_times[-1]},)

            if (i % 10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: ' + str(i - 10), fps=4, format="gif"),
                               "Episode": i})

            # if i % config.save_every == 0:
            #     save(config, save_name="IQL", model=agent.actor_local, wandb=wandb, ep=0)
    print(f"dataloaders preparation time is {dataloaders_preparation_time}")

if __name__ == "__main__":
    print(torch.cuda.get_device_name(torch.cuda.current_device()))
    config = get_config()
    train(config)
