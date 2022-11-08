"""
A quick script to run a sanity check on all environments.
"""
import gym
import d4rl
import d4rl.kitchen
import d4rl.flow

ENVS = [
    'maze2d-open-v0',
    'maze2d-umaze-v1',
    'maze2d-medium-v1',
    'maze2d-large-v1',
    'maze2d-open-dense-v0',
    'maze2d-umaze-dense-v1',
    'maze2d-medium-dense-v1',
    'maze2d-large-dense-v1',
    'minigrid-fourrooms-v0',
    'minigrid-fourrooms-random-v0',
    'pen-human-v0',
    'pen-cloned-v0',
    'pen-expert-v0',
    'hammer-human-v0',
    'hammer-cloned-v0',
    'hammer-expert-v0',
    'relocate-human-v0',
    'relocate-cloned-v0',
    'relocate-expert-v0',
    'door-human-v0',
    'door-cloned-v0',
    'door-expert-v0',
    'halfcheetah-random-v0',
    'halfcheetah-medium-v0',
    'halfcheetah-expert-v0',
    'halfcheetah-medium-replay-v0',
    'halfcheetah-medium-expert-v0',
    'walker2d-random-v0',
    'walker2d-medium-v0',
    'walker2d-expert-v0',
    'walker2d-medium-replay-v0',
    'walker2d-medium-expert-v0',
    'hopper-random-v0',
    'hopper-medium-v0',
    'hopper-expert-v0',
    'hopper-medium-replay-v0',
    'hopper-medium-expert-v0',
    'antmaze-umaze-v0',
    'antmaze-umaze-diverse-v0',
    'antmaze-medium-play-v0',
    'antmaze-medium-diverse-v0',
    'antmaze-large-play-v0',
    'antmaze-large-diverse-v0',
    'kitchen-complete-v0',
    'kitchen-partial-v0',
    'kitchen-mixed-v0',
]

if __name__ == '__main__':
    for env_name in ENVS:
        #print('Checking', env_name)
        env = gym.make(env_name)
        #print('\''+env_name+'\'', ':', '\''+env.dataset_url+'\'')
        print('\''+env_name+'\'', ':', env.ref_min_score, ',')
