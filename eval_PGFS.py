from gym_PGFS.valuation.comparison_vs_random import compare_against_random, draw_comparison_figure
from gym_PGFS.envs.PGFS_trials import gym_PGFS_basic_from_config
from gym_PGFS.config import load_config
from sys import argv

assert len(argv) == 6, "Requires size positional arguments:" \
                       " data_path, agent_state_path, config_path, n_samples, out_image"

transforms = {
    'norm': 'normalized ',
    'scale': 'scaled ',
    'none': ''
}

data_path = argv[1]
agent_path = argv[2]   # 'data/agent_checkpoints/agent_23000.state'
config = argv[3]    # "./gym_PGFS/configs/config_server_default.yaml"
n_samples = int(argv[4])
out_image = argv[5]

conf = load_config(config)  # "./gym_PGFS/configs/config_server_default.yaml"
run_conf = conf['run']
conf['env']['fmodel_kwargs']['fmodel_start_conditions']['train_mode'] = False
env = gym_PGFS_basic_from_config(data_path, conf)  # './data'

ran, ag = compare_against_random(env, agent_path, run_conf, n_samples=n_samples)
transform = transforms[conf['env']['scoring_transform']]
f = draw_comparison_figure(ag, ran, ylabel=transform+conf['env']['scoring'])
f.savefig(out_image)
