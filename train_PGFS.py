from gym_PGFS.rl.runner import Runner
from gym_PGFS.envs.PGFS_trials import gym_PGFS_basic_from_config
from gym_PGFS.config import load_config

from sys import argv
# import os
# import pickle
# import yaml
# from gym_PGFS.utils import ensure_dir_exists

assert len(argv) == 3, "Must specify exactly 2 cmd arguments: root_directory and config address."

root_dir = argv[1]
config = argv[2]
conf = load_config(config)  # "./gym_PGFS/configs/config_server_default.yaml"
env = gym_PGFS_basic_from_config(root_dir, conf)  # './data'

r = Runner(env, conf['run'])
r.env.rmodel.cw.save_checkpoint()

# try:
r.run()
# except BaseException as e:
#     print(f"Exception caught: {str(e)}\nDumping the buffer and other state files...")
# finally:
#     # error or not, save the training state
#     statedir = os.path.join(root_dir, 'state_dumped')
#     ensure_dir_exists(statedir)
#
#     # save the buffer
#     with open(os.path.join(statedir, 'buffer'), 'wb') as f:
#         pickle.dump(r.replay, f)
#
#     # the agent
#     with open(os.path.join(statedir, 'last_agent'), 'wb') as f:
#         pickle.dump(r.agent, f)
#
#     # and the holy spirit
#     with open(os.path.join(statedir, 'last_agent'), 'wb') as f:
#         tracker = r.tracker
#         tracker.close()
#         pickle.dump(tracker, f)
