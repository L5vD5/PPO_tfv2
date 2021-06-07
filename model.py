# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions.normal import Normal
# from torch.distributions.categorical import Categorical
import tensorflow as tf

from gym.spaces import Box, Discrete

##### Model construction #####
def mlp(odim=24, hdims=[256,256], actv=tf.nn.relu(), output_actv=None):
    layers = tf.keras.Sequential;
    for hdim in hdims[:-1]:
        layers.add(tf.keras.layers.Dense(hdim, use_bias=True))
        layers.add(actv)
    layers.append(tf.keras.layers.Dense(hdims[-1]))
    if output_actv is None:
        return layers
    else:
        layers.add(output_actv)
        return layers

class CategoricalPolicy(tf.Module):
    def __init__(self, odim, adim, hdims=[64,64], actv=tf.nn.relu(), output_actv=None):
        super(CategoricalPolicy, self).__init__()
        self.output_actv = output_actv
        self.net = mlp(odim, hdims=hdims, actv=actv, output_actv=output_actv)
        self.logits = tf.keras.layers.Dense(in_features=hdims[-1], out_features=adim)
    def forward(self, x, a=None):
        output = self.net(x)
        logits = self.logits(output)
        if self.output_actv:
            logits = self.output_actv(logits)
        prob = F.softmax(logits, dim=-1)
        dist = Categorical(probs=prob)
        pi = dist.sample()
        logp_pi = dist.log_prob(pi)
        logp = dist.log_prob(a)
        return pi, logp, logp_pi, pi

class GaussianPolicy(tf.Module):    # def mlp_gaussian_policy
    def __init__(self, odim, adim, hdims=[64,64], actv=tf.nn.relu(), output_actv=None):
        super(GaussianPolicy, self).__init__()
        self.output_actv = output_actv
        self.mu = mlp(odim, hdims=hdims+[adim], actv=actv, output_actv=output_actv)
        self.log_std = nn.Parameter(-0.5*torch.ones(adim))
    def forward(self, x, a=None):
        mu = self.mu(x)
        std = self.log_std.exp()
        policy = Normal(mu, std)
        pi = policy.sample()
        # gaussian likelihood
        logp_pi = policy.log_prob(pi).sum(dim=1)
        if a is not None:
            logp = policy.log_prob(a).sum(dim=1)
        else:
            logp = None
        return pi, logp, logp_pi, mu        # 순서 ActorCritic return 값이랑 맞춤.

class ActorCritic(nn.Module):   # def mlp_actor_critic
    def __init__(self, odim, adim, hdims=[64,64], actv=tf.nn.relu(),
                 output_actv=None, policy=None, action_space=None):
        super(ActorCritic,self).__init__()
        if policy is None and isinstance(action_space, Box):
            self.policy = GaussianPolicy(odim, adim, hdims, actv, output_actv)
        elif policy is None and isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(odim, adim, hdims, actv, output_actv)
        self.vf_mlp = mlp(odim, hdims=hdims+[1],
                          actv=actv, output_actv=output_actv)
    def forward(self, x, a=None):
        pi, logp, logp_pi, mu = self.policy(x, a)
        v = self.vf_mlp(x)
        return pi, logp, logp_pi, v, mu