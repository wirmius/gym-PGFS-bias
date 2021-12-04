from gym_PGFS.rl.runner import Runner
from gym_PGFS.envs.PGFS_trials import gym_PGFS_basic_from_config
from gym_PGFS.config import load_config

from sys import argv

assert len(argv) == 3, "Must specify exactly 2 cmd arguments: root_directory and config address."

root_dir = argv[1]
config = argv[2]
conf = load_config(config)  # "./gym_PGFS/configs/config_server_default.yaml"
env = gym_PGFS_basic_from_config(root_dir, conf)  # './data'

r = Runner(env, conf['run'])
r.env.rmodel.cw.save_checkpoint()

r.run()