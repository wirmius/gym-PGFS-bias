import gym
import numpy as np
import torch

from typing import Dict, Generator
from tqdm import tqdm
import os

from ..envs.PGFS_env import PGFS_env
from ..utils import print_verbose, NoKekulizedProducts, ReactionFailed, ProbablyBadHeteroatomsError
from .agents import PGFS_agent
from .rlutils import ReplayBuffer, Transition
from .recorder import MultiTrack
from torch.utils.tensorboard import SummaryWriter

# R T_ a T
class Runner(object):

    def __init__(self,
                 env: PGFS_env,
                 run_config: Dict
                 ):
        # set the configuration variables directly
        for k, v in run_config.items():
            setattr(self, k, v)

        self.env = env

        # initialize the replay buffer
        self.replay = ReplayBuffer(
            max_size=self.max_buffer_size,
            random_state=env.rng        # use the same rng for everything
        )

        # initialise the agent
        self.agent = PGFS_agent(env.action_space,
                                env.observation_space,
                                random_state=env.rng,
                                **run_config,
                                gumbel_tau = run_config['g_tau_start'])   # supply all the arguments to be used by the agent as needed

        self.tracker = None

        # TODO: improve serialization of the agent
        if self.resume_training:
            print_verbose(f"Resuming checkpoint {self.resume_file_name}...", self.verbosity, 1)
            self.agent.load_checkpoint(os.path.join(self.agent_checkpoint_dir, self.resume_file_name))

    @staticmethod
    def _sample_generator(env, agent, buffer, tracker: MultiTrack, gumbel_tau, gamma = 0.99):
        '''
        A generator that makes one sample from the environment at a time with handling of the exceptions and logging.
        Parameters
        ----------
        env
        agent
        buffer
        tracker
        gamma
        gumbel_tau
            either a value or a generator

        Returns
        -------
            Number of transitions generated
        '''
        while True:
            cur_discount = 1
            cur_return = 0
            episode = []
            o = env.reset()
            done = False
            while not done:
                gum_tau_value = next(gumbel_tau) if isinstance(gumbel_tau, Generator) else gumbel_tau
                if agent:
                    # print(gumbel_tau)
                    # print(next(gumbel_tau) if isinstance(gumbel_tau, Generator) else gumbel_tau)
                    action = agent.act(*o,
                                       gumbel_tau = gum_tau_value,
                                       add_noise=True)
                else:
                    action = env.suggest_action()
                try:
                    new_o, r, done, info = env.step( (action[0], action[1]) )
                except (NoKekulizedProducts, ReactionFailed, ProbablyBadHeteroatomsError) as kek:
                    tracker.record_error(kek)
                    break

                # we dont add episodes that end up with no reaction possibilities because that breaks the training algorithm
                if new_o[1].sum() != 0:
                    episode.append(Transition(R_old=o[0], T_mask=o[1],
                                              a=action[0], T=action[1],
                                              reward=r*cur_discount, R_new=new_o[0], T_mask_new=new_o[1]))

                    # update the running characteristics
                    cur_discount *= gamma
                    cur_return += r*cur_discount

                    # track the transition
                    tracker.record_transition(r, action[1], mol_str_rep=str(info['new_molecule']))

                    o = new_o
                    yield gum_tau_value
                else:
                    tracker.record_error("no_rxn")

            tracker.record_episode(cur_return, len(buffer))

            # only fill up the buffer if the episode has been successful
            for trans in episode:
                buffer.add(trans)

    def run(self):
        # prepare the agent for learning
        if self.agent.deployed:
            self.agent.retract()
        # deploy the agent to the GPU
        acopt, cropt = self.agent.deploy(self.lr_actor, self.lr_critic, self.device)

        # STEP 1: fill the buffer with random samples to a certain extent
        # initialize a summarywriter for the random sampling
        preheat_tracker = MultiTrack(
            tensorboard_dir=os.path.join(self.env.rmodel.cw.DATA_DIR, 'tensorboard_random_presample'),
            max_episodes=self.min_buffer_content//self.env.max_steps,
            env=self.env,
            render_dir=os.path.abspath(os.path.join(self.env.rmodel.cw.DATA_DIR, 'bestiary_renders_random')),
            log_prefix='random_sampling'
        )

        # initialize the environment sampler
        random_sampler = Runner._sample_generator(self.env,
                                                  None,
                                                  self.replay,
                                                  preheat_tracker,
                                                  self.gamma)

        # check if we have the minimal number of samples in the buffer already
        if self.replay.counter < self.min_buffer_content:
            print_verbose(f"Must fill up the buffer before beginning the run...", 2, self.verbosity)
            # run sampler until we have enough episodes in the buffer to start learning
            while len(self.replay) < self.min_buffer_content:
                next(random_sampler)

        preheat_tracker.close()

        # STEP 2: proceed with PGFS training routine
        # initialize the tracker for the run
        if not self.tracker:
            self.tracker = MultiTrack(
                tensorboard_dir=os.path.join(self.env.rmodel.cw.DATA_DIR, 'tensorboard'),
                max_episodes=self.max_episodes,
                agent_checkpoint_dir=os.path.abspath(os.path.join(self.env.rmodel.cw.DATA_DIR, 'agent_checkpoints')),
                agent=self.agent,
                checkpoint_every=self.checkpoint_every,
                env=self.env,
                render_dir=os.path.abspath(os.path.join(self.env.rmodel.cw.DATA_DIR, 'bestiary_renders'))
            )

        # initialize the environment sampler
        sampler = Runner._sample_generator(self.env,
                                           self.agent,
                                           self.replay,
                                           self.tracker,
                                           exp_decay_annealing(self.g_tau_start, self.g_tau_end,
                                                               self.max_episodes * self.env.max_steps / 2, spacing=1000),
                                           self.gamma,
                                           )

        # initialize the gumbel tau decay iterator (make sure to leave a bit of spacing
        # and account for the fact, that gumbel_tau changes every transition)
        #

        # now we should have a buffer that is at the minimum capacity at least
        # can start going one episode at a time
        while self.tracker.total_episodes < self.max_episodes:
            # run the env under the policy
            gumbel_tau = next(sampler)

            # sample a minibatch
            mb = self.replay.sample(self.batch_size)
            # update the agent on the minibatch
            self.agent.train_DDPG(mb,
                                  cropt,
                                  acopt,
                                  self.target_tau,
                                  self.gamma,
                                  tracker=self.tracker,
                                  gumbel_tau=gumbel_tau)

        # take the agent back to the CPU
        self.agent.retract()
        self.tracker.close()


def linear_annealing(low, high, timespan:int, spacing = 0):
    for i in range(spacing):
        yield low
    for i in range(timespan):
        yield low + (high-low)*i/timespan
    while True:
        yield high


def exp_decay_annealing(start, end, half_life:int, spacing = 0):
    for i in range(spacing):
        yield start
    tau = half_life/np.log(2)
    i = 0
    while True:
        yield end + (start-end)*np.exp(-i/tau)
        i += 1
