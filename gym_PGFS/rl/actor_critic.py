import torch
import torch.nn as nn
import torch.nn.functional as F

from copy import deepcopy

from typing import Dict, List

class ActorCritic(nn.Module):
    '''
    Implements D3PG actor-critic for continuous action spaces.
    Including the target networks.
    '''
    def __init__(self,
                 len_obs: int,
                 len_templates: int,
                 len_act: int,
                 accr_config: Dict):
        super(ActorCritic, self).__init__()

        # set the basic parameters
        self.len_obs = len_obs
        self.len_templates = len_templates
        self.len_act = len_act

        # get the actors and the critics
        ac_config = accr_config['actor']
        cr_config = accr_config['critic']

        # initialize the actor and the critic
        self.actor = Actor(self.len_obs,
                           self.len_templates,
                           self.len_act,
                           **ac_config
                           )

        self.critic = Critic(self.len_obs,
                             self.len_templates,
                             self.len_act,
                             **cr_config
                             )

        # initialize the target networks
        self.target_actor = Actor.from_actor(self.actor)
        self.target_criticA = Critic.from_critic(self.critic)
        self.target_criticB = Critic.from_critic(self.critic)

        # disable gradient for the target networks
        self.target_actor.requires_grad_(False)
        self.target_criticA.requires_grad_(False)
        self.target_criticB.requires_grad_(False)

        # for soft updating
        self.soft_update_counter = 0
        self.update_A = accr_config['update_tA_every']
        self.update_B = accr_config['update_tB_every']
        self.update_Actor = accr_config['update_tAc_every']

    def soft_target_nets_update(self, tau_critic, tau_actor):
        self.soft_update_counter += 1
        if self.soft_update_counter % self.update_A:
            soft_target_update(self.critic, self.target_criticA, tau_critic)
        if self.soft_update_counter % self.update_B:
            soft_target_update(self.critic, self.target_criticB, tau_critic)
        if self.soft_update_counter % self.update_Actor:
            soft_target_update(self.actor, self.target_actor, tau_actor)


@torch.no_grad()
def soft_target_update(netfrom: nn.Module, netto: nn.Module, tau: float):
    for key in netfrom.state_dict().keys():
        netto.state_dict()[key] = (1-tau)*netto.state_dict()[key] + tau*netfrom.state_dict()[key]


class Actor(nn.Module):
    def __init__(self,
                 len_obs,
                 len_templates,
                 len_act,
                 smooth_sigma,
                 smooth_c,
                 f_net_layers,
                 pi_net_layers):
        super(Actor, self).__init__()

        self.f = f_network(len_obs,
                           len_templates,
                           f_net_layers)
        self.pi = pi_network(len_obs,
                             len_templates,
                             len_act,
                             pi_net_layers)

        self.len_obs = len_obs
        self.len_templates = len_templates
        self.len_act = len_act
        self.smooth_sigma = smooth_sigma
        self.smooth_c = smooth_c

    @classmethod
    def from_actor(cls, actor):
        '''copyact = cls(actor.len_obs,
                      actor.len_templates,
                      actor.len_act,
                      actor.gumbel_tau,
                      actor.smooth_sigma,
                      actor.smooth_c)
        src_state_dict = actor.state_dict().deepcopy()'''
        return deepcopy(actor)

    @torch.no_grad()
    def exploration_noise(self, array_target):
        return torch.clip_(torch.randn(*array_target.shape).to(array_target.device) * self.smooth_sigma,
                           min=-self.smooth_c, max=self.smooth_c)

    @staticmethod
    def _actor(mol: torch.Tensor,
               T_mask: torch.Tensor,
               f: nn.Module,
               pi: nn.Module,
               gumbel_tau: float):

        # compute the template
        T_o = f(mol)
        T = T_o * T_mask
        # at this point the gradient of the zero entries is zero, so we can do a very dirty trick in order to
        # ensure that gumbel softmax never picks the wrong templates
        T = T + (T_mask-1)*10000
        # the gradient should not flow through the -10000k cells since they have been multiplied by zero before
        # so this operation should not mess up the f networks gradients but effectively ensure the correct
        # gumbel_softmax outputs

        # apply gumbel softmax
        T = F.gumbel_softmax(T, tau=gumbel_tau, hard=True)

        # compute the second reagent
        a = pi(mol, T)

        return a, T

    def forward(self, *obs, gumbel_tau):
        return Actor._actor(
            obs[0],
            obs[1],
            self.f,
            self.pi,
            gumbel_tau
        )


class Critic(nn.Module):
    def __init__(self,
                 len_obs,
                 len_templates,
                 len_act,
                 q_net_layers,
                 **kwargs,
                 ):
        super(Critic, self).__init__()
        self.len_obs = len_obs
        self.len_templates = len_templates
        self.len_act = len_act
        self.Q = q_network(self.len_obs,
                           self.len_templates,
                           self.len_act,
                           q_net_layers,
                           )

    @classmethod
    def from_critic(cls, critic):
        return deepcopy(critic)

    @staticmethod
    def _critic(mol: torch.Tensor,
                T: torch.Tensor,
                action: torch.Tensor,
                q: nn.Module):
        return q(mol, action, T)

    def forward(self, mols, Ts, actions):
        return Critic._critic(mols,
                              Ts,
                              actions,
                              self.Q)

class pi_network(nn.Module):
    def __init__(self,
                 len_observation_vec: int,
                 len_templates: int,
                 len_action_vec: int,
                 pi_net_layers: List[int]
                 ):
        super(pi_network, self).__init__()
        # len_in, 256, 256, 167
        self.net = _build_sequential_linear(len_observation_vec+len_templates,
                                            len_action_vec,
                                            pi_net_layers,
                                            nn.ReLU()
                                            )

    def forward(self, obs, temp):
        # input assumed to be [B, N], where N is the feature size, therefore we concatenate the two to obtain the stuff
        if obs.ndim > 1 and temp.ndim > 1:
            net_input = torch.cat((obs, temp), dim=1)
        else:
            net_input = torch.cat((obs, temp), dim=0)
        # the output (tanh applied then multiplied by 3 - the rlv descriptors are normalized, 3 stds should
        # cover 99% of compounds while maintaining some sort of constraints
        return torch.tanh(self.net(net_input))*3


class f_network(nn.Module):
    def __init__(self,
                 len_observation_vec: int,
                 len_templates: int,
                 f_net_layers: List[int],
                 ):
        super(f_network, self).__init__()
        # paper: 256, 128, 128, ReLU; tanh output again?
        self.net = _build_sequential_linear(len_observation_vec,
                                            len_templates,
                                            f_net_layers,
                                            nn.ReLU()
                                            )

    def forward(self, obs):
        # tanh constraint to improve resilience against modus collapse
        return torch.tanh(self.net(obs))


class q_network(nn.Module):
    def __init__(self,
                 len_observation_vec: int,
                 len_templates: int,
                 len_action_vec: int,
                 q_net_layers: List[int]
                 ):
        super(q_network, self).__init__()
        # paper: 256, 64, 16, ReLU;
        self.net = _build_sequential_linear(len_observation_vec+len_templates+len_action_vec,
                                            1,
                                            q_net_layers,
                                            nn.ReLU()
                                            )

    def forward(self, obs, action, temp):
        # I believe the template should be that produced by the f network
        # input assumed to be [B, N], where N is the feature size, therefore we concatenate the two to obtain the stuff
        if obs.ndim > 1 and temp.ndim > 1:
            net_input = torch.cat((obs, action, temp), dim=1)
        else:
            net_input = torch.cat((obs, action, temp), dim=0)
        return self.net(net_input)


def _build_sequential_linear(input_size, output_size, layer_list: List[int], activation):
    mlist = nn.ModuleList()
    l_prev = input_size
    for l in layer_list:
        mlist.append(nn.Linear(in_features=l_prev, out_features=l))
        mlist.append(activation)
        l_prev = l
    # add the output layer
    mlist.append(nn.Linear(in_features=l_prev, out_features=output_size))
    # return as a sequential
    return nn.Sequential(*mlist)