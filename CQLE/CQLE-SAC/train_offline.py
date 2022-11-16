import gym
import d4rl
import numpy as np
from collections import deque
import torch
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
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
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

    datasets = split_dataset(
        tensors["observations"],
        tensors["actions"],
        tensors["rewards"][:, None],
        tensors["next_observations"],
        tensors["terminals"][:, None],
        num_datasets=config.num_agents,
        is_GMM=config.is_GMM,
        s=config.s
    )

    dataloaders = []
    for i in range(config.num_agents):
        tensordata = TensorDataset(datasets[i]["observations"],
                                   datasets[i]["actions"],
                                   datasets[i]["rewards"],
                                   datasets[i]["next_observations"],
                                   datasets[i]["terminals"])
        dataloaders.append(DataLoader(tensordata, batch_size=config.batch_size, shuffle=True))

    print("------------------------------------dataloaders generated------------------------------")

    return dataloaders, eval_env

def evaluate(env, policy, eval_runs=5): 
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

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

    dataloaders, env = prep_dataloaders(config)
    # env.action_space.seed(config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batches = 0
    average10 = deque(maxlen=10)
    
    with wandb.init(project="CQL-ensemble-offline", name=config.run_name, config=config):
        
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
            is_GMM=config.is_GMM==1,
            s=config.s,
        )

        # wandb.watch(ensemble, log="gradients", log_freq=10)
        if config.log_video:
            env = gym.wrappers.Monitor(env, './video', video_callable=lambda x: x%10==0, force=True)

        eval_reward = evaluate(env, ensemble)
        wandb.log({"Test Reward": eval_reward, "Episode": 0, "Batches": batches}, step=batches)
        for i in range(1, config.episodes+1):

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
                    policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = ensemble.CQL_agents[agent_id].learn((states, actions, rewards, next_states, dones))
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
                eval_reward = evaluate(env, ensemble)
                wandb.log({"Test Reward": eval_reward, "Episode": i, "Batches": batches}, step=batches)

                average10.append(eval_reward)
                print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {}".format(i, eval_reward, policy_loss, batches,))
            
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
                       "Episode": i})

            if (i %10 == 0) and config.log_video:
                mp4list = glob.glob('video/*.mp4')
                if len(mp4list) > 1:
                    mp4 = mp4list[-2]
                    wandb.log({"gameplays": wandb.Video(mp4, caption='episode: '+str(i-10), fps=4, format="gif"), "Episode": i})

            # if i % config.save_every == 0:
            #     save(config, save_name="IQL", model=agent.actor_local, wandb=wandb, ep=0)

if __name__ == "__main__":
    config = get_config()
    train(config)
