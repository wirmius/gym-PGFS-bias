from gym_PGFS.chemutils import ChemWorld
from gym_PGFS.envs.PGFS_env import PGFS_env
from gym_PGFS.envs.PGFS_goal_env import PGFS_Goal_env
from gym_PGFS.function_set_basic import get_forward_model
from gym_PGFS.scorers.scorer import get_scoring_function
from typing import Union, Dict
import os

from gym_PGFS.config import load_config


class gym_PGFS_goal(PGFS_Goal_env):
    def __init__(self, cw=None):
        super().__init__(
            give_info=True,
            max_steps=10,
            enable_render=True,
            cw=ChemWorld() if not cw else cw
        )


class gym_PGFS_basic_from_config(PGFS_env):
    def __init__(self,
                 cw_dir: Union[str, os.PathLike] = '',
                 config: Union[Dict, str, os.PathLike] = None):
        if isinstance(config, str) or isinstance(config, os.PathLike) or config is None:
            config = load_config(config)

        env_config = config['env']

        scoring_config = env_config['scoring']
        # initialize the scoring function
        scoring_fn = get_scoring_function(**scoring_config)

        give_info = env_config['give_info']
        render = env_config['render']
        max_steps = env_config['max_steps']
        fmodel_type = env_config['fmodel_type']
        fmodel_kwargs = env_config['fmodel_kwargs']

        cw = ChemWorld(comp_dir=cw_dir, config=config)
        cw.deploy()

        super().__init__(
            scoring_fn=scoring_fn,
            give_info=give_info,
            max_steps=max_steps,
            render=render,
            fmodel=get_forward_model(fmodel_type),
            cw=cw,
            **fmodel_kwargs
        )
