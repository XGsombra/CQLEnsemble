from gym.envs.registration import register
from d4rl.gym_bullet import gym_envs

register(
    id='bullet-hopper-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_hopper_env',
    max_episode_steps=1000,
)

register(
    id='bullet-halfcheetah-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_cheetah_env',
    max_episode_steps=1000,
)

register(
    id='bullet-ant-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_ant_env',
    max_episode_steps=1000,
)

register(
    id='bullet-walker2d-v0',
    entry_point='d4rl.gym_bullet.gym_envs:get_walker_env',
    max_episode_steps=1000,
)

