from gym.envs.registration import register

register(
    id='gym-PGFS-basic-v0',
    entry_point='gym_PGFS.envs.PGFS_trials:gym_PGFS_basic',
)

register(
    id='gym-PGFS-basic-v1',
    entry_point='gym_PGFS.envs.PGFS_trials:gym_PGFS_basic',
)
