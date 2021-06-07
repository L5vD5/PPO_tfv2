import datetime, gym, os, pybullet_envs, psutil, time, os
import scipy.signal
import numpy as np
import tensorflow as tf
import datetime,gym,os,pybullet_envs,time,psutil,ray
from Replaybuffer import PPOBuffer
import random
from config import Config

print ("Packaged loaded. TF version is [%s]."%(tf.__version__))

class Agent(object):
    def __init__(self):
        # Config
        self.config = Config()

        # Environment
        self.env, self.eval_env = get_envs()
        odim = self.env.observation_space.shape[0]
        adim = self.env.action_space.shape[0]

        # Actor-critic model
        ac_kwargs = dict()
        ac_kwargs['action_space'] = self.env.action_space
        self.actor_critic = ActorCritic(odim, adim, self.config.hdims,**ac_kwargs)
        self.buf = PPOBuffer(odim=self.odim,adim=adim,size=self.config.steps_per_epoch,
                             gamma=self.config.gamma,lam=self.config.lam)

        # Buffer (Memory)
        self.buffer = PPOBuffer(size=self.config.buffer_size, odim=self.odim, adim=self.adim, gamma=self.config.gamma,lam=self.config.lam)

        # Optimizers
        self.train_pi = tf.keras.optimizers.Adam(learning_rate=self.config.pi_lr)
        self.train_v = tf.keras.optimizers.Adam(learning_rate=self.config.vf_lr)
        # self.train_pi = torch.optim.Adam(self.actor_critic.policy.parameters(), lr=self.config.pi_lr)
        # self.train_v = torch.optim.Adam(self.actor_critic.vf_mlp.parameters(), lr=self.config.vf_lr)


    def train(self):
        start_time = time.time()

        o, r, d, ep_ret, ep_len, n_env_step = self.env.reset(), 0, False, 0, 0, 0
        for epoch in range(self.config.epochs):
            o, d, ep_ret, ep_len = self.env.reset(), False, 0, 0
            while not(d):
                a = self.env.action_space.sample()
                o1, r, d, _ = self.env.step(a)


                ep_len += 1
                ep_ret += r
                n_env_step += 1
                # Save the Experience to our buffer
                self.buffer.append(o, a, r, o1, d)
                o = o1
            # Evaluate
            if (epoch == 0) or (((epoch + 1) % self.config.evaluate_every) == 0):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, self.config.epochs, epoch / self.config.epochs * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                _ = self.eval_env.render(mode='human')
                while not (d or (ep_len == self.config.steps_per_epoch)):
                    # a = sess.run(model['mu'], feed_dict={model['o_ph']: o.reshape(1, -1)})
                    o, r, d, _ = self.eval_env.step(a)
                    _ = self.eval_env.render(mode='human')
                    ep_ret += r  # compute return
                    ep_len += 1
                print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))

def get_envs():
    env_name = 'AntBulletEnv-v0'
    env,eval_env = gym.make(env_name),gym.make(env_name)
    _ = eval_env.render(mode='human') # enable rendering on test_env
    _ = eval_env.reset()
    for _ in range(3): # dummy run for proper rendering
        a = eval_env.action_space.sample()
        o,r,d,_ = eval_env.step(a)
        time.sleep(0.01)
    return env,eval_env

a = Agent()
a.train()
