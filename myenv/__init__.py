from gymnasium.envs.registration import register

register(
    id='NoisyGridworld-v0',
    entry_point='myenv.envs:NoisyGridworldEnv',
    max_episode_steps=20,
)

register(
    id='NoisyLimitCycle-v0',
    entry_point='myenv.envs:NoisyLimitCycleEnv',
    max_episode_steps=200,
)
