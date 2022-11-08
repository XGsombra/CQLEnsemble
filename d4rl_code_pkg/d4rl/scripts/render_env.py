import gym
import d4rl
import d4rl.kitchen
import numpy as np

ENVS = [
    'mini-kitchen-microwave-kettle-light-slider-v0',
    'kitchen-microwave-kettle-light-slider-v0',
    'kitchen-microwave-kettle-bottomburner-light-v0',
]

if __name__ == "__main__":
    for env_name in ENVS:
        env = gym.make(env_name)
        dset = env.get_dataset()
        env.reset()
        env.step(np.zeros_like(env.action_space.sample()))
        for _ in range(10000):
            env.render()
        break
        #img = env.render(mode='rgb_array')

