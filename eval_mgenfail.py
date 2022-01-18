import os

from gym_PGFS.valuation.comparison_vs_random import compare_mgenfail_mode, draw_mgen_comparison_figure
from gym_PGFS.envs.PGFS_trials import gym_PGFS_basic_from_config
from gym_PGFS.config import load_config
from sys import argv

assert len(argv) == 6, "Requires size positional arguments:" \
                       " data_path, agent_state_path, config_path, n_samples, out_image"

# transforms = {
#     'norm': 'normalized ',
#     'scale': 'scaled ',
#     'none': ''
# }

data_path = argv[1]
agent_path = argv[2]   # 'data/agent_checkpoints/agent_23000.state'
config = argv[3]    # "./gym_PGFS/configs/config_server_default.yaml"
n_samples = int(argv[4])
out_image = argv[5]

conf = load_config(config)  # "./gym_PGFS/configs/config_server_default.yaml"
run_conf = conf['run']
env = gym_PGFS_basic_from_config(data_path, conf)  # './data'

os_score, mcs_score, dcs_score, random_scores = compare_mgenfail_mode(env, agent_path, run_conf, n_samples=n_samples)
# transform = transforms[conf['env']['scoring_transform']]
f = draw_mgen_comparison_figure(os_score,
                                mcs_score,
                                dcs_score,
                                random_scores,
                                ylabel=conf['env']['scoring']['name'],
                                xlabel=f"step #{agent_path.split('/')[-1].split('.')[0].split('_')[-1]}")
f.savefig(out_image)
