import os

import numpy as np

import torch
from torch import nn
import torch.optim as opt
from torch.utils.tensorboard import SummaryWriter

from typing import Union

from gym.spaces import Space, Tuple

from .rlutils import Transition, ReplayBuffer
from .actor_critic import ActorCritic
from .recorder import MultiTrack
from ..utils import np_to_onehot


class PGFS_agent(object):
    def __init__(self,
                 actions_space: Space,
                 observation_space: Space,
                 random_state: np.random.RandomState = None,
                 **kwargs
                 ):

        # set the configuration variables directly to be class variables
        for k, v in kwargs.items():
            setattr(self, k, v)

        # some assertions to avoid obvious errors
        assert isinstance(actions_space, Tuple)
        assert isinstance(observation_space, Tuple)
        assert observation_space[1].shape[0] == actions_space[1].n

        # fix the random state
        self.random = random_state

        # parametrize the model
        len_templates = actions_space[1].n  # the number of templates to deal with
        len_obs = observation_space[0].shape[0]
        len_act = actions_space[0].shape[0]

        self.ac = ActorCritic(len_obs=len_obs,
                              len_templates=len_templates,
                              len_act=len_act,
                              accr_config=self.actor_critic
                              )

        self.deployed = False
        self.last_gumbel_tau = self.gumbel_tau_default

    @torch.no_grad()
    def act(self, molecule, T_mask, gumbel_tau = None, add_noise = False):
        device = next(self.ac.actor.parameters()).device

        mol = torch.from_numpy(molecule).to(device).type(torch.float32)
        T_m = torch.from_numpy(T_mask).to(device).type(torch.float32)

        if not gumbel_tau:
            gumbel_tau = self.last_gumbel_tau
        action, T_probs = self.ac.actor(mol, T_m, gumbel_tau=gumbel_tau)
        if add_noise:
            action += self.ac.actor.exploration_noise(action)

        T_discrete = torch.argmax(T_probs).detach().cpu().numpy().item()

        a = action.detach().cpu().numpy()

        # template is returned as a discrete value because that is what the environment takes
        return a, T_discrete

    def deploy(self,
               alpha_actors: float,
               alpha_critic: float,
               device=None,
               ):
        # move the networks to the device
        if device:
            self.ac = self.ac.to(device)

        # initialize the optimisers
        actor_opt = opt.Adam(self.ac.actor.parameters(), lr=alpha_actors)
        critic_opt = opt.Adam(self.ac.critic.parameters(), lr=alpha_critic)

        self.deployed = True

        return actor_opt, critic_opt

    def retract(self):
        device = 'cpu'
        # move the networks to the device
        self.ac.zero_grad()
        self.ac = self.ac.to(device)

        self.deployed = False

    def train_DDPG(self,
                   batch: Transition,
                   critic_opt,
                   actor_opt,
                   target_tau: int,
                   gamma: float,
                   tracker: MultiTrack,
                   gumbel_tau: float,
                   ):
        # get the f network cross entropy coefficient
        f_net_ce_modifier = self.f_net_ce_modifier

        device = next(self.ac.actor.parameters()).device
        critic_opt.zero_grad()
        actor_opt.zero_grad()

        # unpack the batch (as numpy arrays)
        states_i = torch.from_numpy(batch.R_old).to(device).type(torch.float32)
        states_iplus = torch.from_numpy(batch.R_new).to(device).type(torch.float32)
        T_masks_i = torch.from_numpy(batch.T_mask).to(device).type(torch.float32)
        # T_taken assumed to be discrete, has to be converted to one hot
        T_taken = torch.from_numpy(np_to_onehot(batch.T, num_classes=self.ac.len_templates)[:, 0, :]).to(device).type(torch.float32)
        # also have the non-encoded index version for loss computation
        T_taken_nenc = torch.flatten(torch.from_numpy(batch.T).to(device).type(torch.long))
        T_masks_iplus = torch.from_numpy(batch.T_mask).to(device).type(torch.float32)
        rewards = torch.from_numpy(batch.reward).to(device).type(torch.float32)
        actions = torch.from_numpy(batch.a).to(device).type(torch.float32)

        # compute the optimization target of the critic
        with torch.no_grad():
            a_accent, T_accent = self.ac.target_actor(states_iplus, T_masks_iplus, gumbel_tau=gumbel_tau)
            # apply smoothing to the target policy only
            a_accent += self.ac.actor.exploration_noise(a_accent)

            # take the lowest q value to improve stability
            qA_accent = self.ac.target_criticA(states_iplus, a_accent, T_accent)
            qB_accent = self.ac.target_criticB(states_iplus, a_accent, T_accent)
            q_accent = torch.min(torch.cat((qA_accent, qB_accent), dim=1))
            y = rewards + gamma*q_accent

        # make an optimisation step on the critic
        q = self.ac.critic(states_i, actions, T_taken)
        critic_loss = nn.MSELoss(reduction='mean').to(device)(q, y)   # since we are seeking to maximise
        critic_loss.backward()
        critic_opt.step()
        critic_opt.zero_grad()

        # now make an optimization step on the policy
        self.ac.critic.requires_grad_(False)   # disable gradient in the critic
        a, T_probs = self.ac.actor(states_i, T_masks_i, gumbel_tau=gumbel_tau)
        # primary loss on the objective - Q function to optimise the parameters of the actor
        actor_loss = -self.ac.critic(states_i, a, T_probs)    # propagate forward with the actor
        # reduce actors loss: with mean in the batch dimension (as it is basically the value of the critic, it is (Nx1)
        actor_loss = torch.mean(actor_loss)
        # secondary loss on the f net as suggested by the paper to improve gradient and convergence in the f net
        f_net_loss = nn.NLLLoss(reduction='mean').to(device)(torch.log(T_probs+1e-7), T_taken_nenc)
        # combine the two losses
        total_loss = actor_loss+f_net_ce_modifier*f_net_loss
        # back propagate
        total_loss.backward()
        actor_opt.step()
        actor_opt.zero_grad()
        self.ac.critic.requires_grad_(True)  # re-enable the gradient in the critic


        # bring the target networks closer to the behaviour networks
        self.ac.soft_target_nets_update(target_tau, target_tau)

        # log some of the information into the tensorboard
        tracker.record_net_update(
            actor_loss.norm(p=1).detach().cpu().numpy().item(),
            critic_loss.norm(p=1).detach().cpu().numpy().item(),
            f_net_loss.norm(p=1).detach().cpu().numpy().item(),
            parameters={
                'training/batch_size': q.shape[0],
                'gumbel_tau': gumbel_tau
            }
        )
        # update to keep act with the fresh value
        self.last_gumbel_tau = gumbel_tau

        return actor_loss.detach().cpu().numpy().item(), critic_loss.detach().cpu().numpy()

    def save_checkpoint(self, fname: Union[str, os.PathLike]):
        torch.save({
            'actor-critic': self.ac
        }, fname)

    def load_checkpoint(self, fname: Union[str, os.PathLike]):
        sdict = torch.load(fname)
        #self.ac.load_state_dict(sdict['actor-critic'])
        # a dirty patch to work around me not paying attention
        self.ac = sdict['actor-critic']

        return
