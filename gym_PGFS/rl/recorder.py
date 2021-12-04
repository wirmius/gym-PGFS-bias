import os
from collections import Counter
from typing import Dict
import datetime

from torch.utils.tensorboard import SummaryWriter

from ..envs.PGFS_env import PGFS_env

from ..utils import shelf_directory, \
    NoKekulizedProducts, ReactionFailed, ProbablyBadHeteroatomsError


class MultiTrack(object):
    '''
    A class for recording data on the whole training process
    '''
    CHECKPOINT_FNAME = "agent_{:4}.state"
    SVG_RENDER_FNAME = "render_ep_{}.svg"
    MESSAGE_STRING = "\rEpisode {ep:02d}: {it:3.0f}% - {current:20.20s}  | Time elapsed: {timestamp:8.8s} | Buffer content: {buff:06d} | Rate of good samples: {good_samples:3.0f}%"

    def __init__(self,
                 tensorboard_dir,
                 max_episodes,
                 agent_checkpoint_dir=None,
                 render_dir=None,
                 agent = None,
                 env: PGFS_env = None,
                 checkpoint_every: int = 5000,
                 log_prefix: str = "training",
                 ):
        shelf_directory(tensorboard_dir)
        self.sw = SummaryWriter(log_dir=tensorboard_dir)
        if not log_prefix.endswith('/'):
            log_prefix += '/'
        self.log_pref = log_prefix

        if agent and agent_checkpoint_dir:
            shelf_directory(agent_checkpoint_dir)
            self.total_updates = 0
            self.CP_DIR = agent_checkpoint_dir
            self.checkpoint_every = checkpoint_every
            self.agent = agent      # if agent is none, no checkpointing is performed
        else:
            self.agent = None

        if render_dir and env:
            shelf_directory(render_dir)
            self.render_dir = render_dir
            self.env = env
            # keep track of the best performing runs
        else:
            self.env = None
        self.best_reward = -1000
        self.render_env_when_done = False

        self.total_episodes = 0
        self.total_transitions = 0
        self.current_episode_len = 0

        # error tracking information
        self.ignored_transitions = 0
        self.failed_episodes = 0

        # different types of problematic episode endings
        self.no_kek_episodes = 0
        self.rxn_fail_episodes = 0
        self.no_rxn_possible = 0

        # some indicators on the replay buffer
        self.last_replay_buffer_len = 0

        # time recording
        self.time_started = datetime.datetime.now()
        self.max_episodes = max_episodes

        # track the reactions chosen by the agent
        self.agents_actions_record = Counter()

    def record_transition(self, reward, template_selected, mol_str_rep=None):
        self.sw.add_scalar(self.log_pref+'env/reward', reward, self.total_transitions)
        self.agents_actions_record.update([template_selected])
        self.current_episode_len+=1
        self.total_transitions+=1
        if self.best_reward <= reward:
            self.sw.add_scalar(self.log_pref + 'env/best_reward', reward, self.total_transitions)
            if mol_str_rep:
                self.sw.add_text(self.log_pref + 'env/best_molecule', mol_str_rep, global_step=self.total_transitions)
            self.best_reward = reward
            self.render_env_when_done = True

    def record_episode(self, ret, replay_buffer_size):
        self.sw.add_scalar(self.log_pref+'env/return', ret, self.total_episodes)
        self.sw.add_scalar(self.log_pref+'env/replay_buffer_length', replay_buffer_size, self.total_episodes)
        self.last_replay_buffer_len = replay_buffer_size
        self.__finish_episode()

    def record_net_update(self,
                          actor_loss,
                          critic_loss,
                          f_net_ce_loss,
                          parameters: Dict):
        self.total_updates +=1

        self.sw.add_scalar(self.log_pref+'networks/critic_loss', critic_loss, global_step=self.total_updates)
        self.sw.add_scalar(self.log_pref+'networks/actor_loss', actor_loss, global_step=self.total_updates)
        self.sw.add_scalar(self.log_pref+'networks/f_net_loss', f_net_ce_loss, global_step=self.total_updates)

        # add some extra parameters
        for k, v in parameters.items():
            self.sw.add_scalar(self.log_pref + 'networks/' + k, v, global_step=self.total_updates)

        if self.agent and self.total_episodes % self.checkpoint_every == 0:
            self.agent.save_checkpoint(os.path.join(self.CP_DIR, MultiTrack.CHECKPOINT_FNAME.format(self.total_episodes)))

    def record_error(self, kek):
        if isinstance(kek, NoKekulizedProducts):
            self.no_kek_episodes += 1
            self.sw.add_text(self.log_pref+'exceptions/reaction', kek.rxn, global_step=self.total_episodes)
            self.sw.add_text(self.log_pref+'exceptions/reactants', str(kek.reactants), global_step=self.total_episodes)
            self.sw.add_text(self.log_pref+'exceptions/products', str(kek.products), global_step=self.total_episodes)
            self.sw.add_text(self.log_pref+'exceptions/reaction_id', str(kek.rxn_id), global_step=self.total_episodes)
            self.sw.add_scalar(self.log_pref + 'env/reactions_failed', self.no_kek_episodes, global_step=self.total_episodes)
        elif isinstance(kek, ReactionFailed):
            self.rxn_fail_episodes += 1
            self.sw.add_text(self.log_pref+'exceptions/reaction', kek.rxn, global_step=self.total_episodes)
            self.sw.add_text(self.log_pref+'exceptions/reactants', str(kek.reactants), global_step=self.total_episodes)
            self.sw.add_text(self.log_pref+'exceptions/reaction_id', str(kek.rxn_id), global_step=self.total_episodes)
            self.sw.add_scalar(self.log_pref + 'env/reactions_failed', self.rxn_fail_episodes, global_step=self.total_episodes)
        elif isinstance(kek, ProbablyBadHeteroatomsError):
            self.sw.add_text(self.log_pref + 'exceptions/strange_compounds', str(kek.smiles), global_step=self.total_episodes)
            self.failed_episodes -= 1
            self.ignored_transitions -= self.current_episode_len
        elif kek=="no_rxn":
            self.no_rxn_possible += 1
            self.sw.add_scalar(self.log_pref + 'env/no_reaction_possible', self.no_rxn_possible, global_step=self.total_episodes)
            # dont count this as a failure
            self.failed_episodes -= 1
            self.ignored_transitions -= self.current_episode_len
        else:
            raise ValueError("Strange error recorded just now: " + str(kek))

        # count the current episode as lost transitions as they are not to be used in the training
        self.failed_episodes += 1
        self.ignored_transitions += self.current_episode_len

        # round up the episode (redundant)
        #self.__finish_episode(current="Environment error encountered")

    def record_custom_string(self, tag: str, content: str):
        self.sw.add_text(self.log_pref+tag, content, global_step=self.total_episodes)

    def __finish_episode(self, current="Training..."):
        if self.render_env_when_done and self.env:
            # render the environment and save it to a file in the specified dir
            with open(os.path.join(self.render_dir, self.SVG_RENDER_FNAME.format(self.total_episodes)), "wt") as f:
                f.write(self.env.render())
        self.render_env_when_done = False
        self.current_episode_len = 0
        self.total_episodes += 1
        self.sw.flush()
        timed = datetime.datetime.now() - self.time_started
        print(
            MultiTrack.MESSAGE_STRING.format(
                ep=self.total_episodes,
                it=100 * self.total_episodes / self.max_episodes,
                current=current,
                timestamp=str(timed),
                buff=self.last_replay_buffer_len,
                good_samples=100 - 100 * self.ignored_transitions / self.total_transitions+1), end=''
        )

    def finalize(self):
        print('\n')
        print(self.agents_actions_record)

    def close(self):
        self.finalize()
        self.sw.close()