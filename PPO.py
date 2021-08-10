import datetime, gym, os, pybullet_envs, psutil, time, os
import scipy.signal
import numpy as np
import tensorflow as tf
import datetime,gym,os,pybullet_envs,time,psutil,ray
from Replaybuffer import PPOBuffer
from model import *
import random
from collections import deque

print ("Packaged loaded. TF version is [%s]."%(tf.__version__))


class Agent(object):
    def __init__(self, args):
        # Config
        self.gamma = args.gamma
        self.lam = args.lam
        self.hdims = args.hdims
        self.steps_per_epoch = args.steps_per_epoch
        self.pi_lr = args.pi_lr
        self.vf_lr = args.vf_lr
        self.train_pi_iters = args.train_pi_iters
        self.clip_ratio = args.clip_ratio
        self.target_kl = args.target_kl
        self.train_v_iters = args.train_v_iters
        self.epochs = args.epochs
        self.print_every = args.print_every
        self.max_ep_len = args.max_ep_len
        self.evaluate_every = args.evaluate_every

        # Environment
        self.env, self.eval_env = get_envs()
        odim = self.env.observation_space.shape[0]
        adim = self.env.action_space.shape[0]

        # Actor-critic model
        ac_kwargs = dict()
        ac_kwargs['action_space'] = self.env.action_space
        self.actor_critic = ActorCritic(odim, adim, self.hdims,**ac_kwargs)
        self.buf = PPOBuffer(odim=odim,adim=adim,
                             size=self.steps_per_epoch, gamma=self.gamma,lam=self.lam)

        # Optimizers
        self.train_pi = tf.keras.optimizers.Adam(learning_rate=self.pi_lr)
        self.train_v = tf.keras.optimizers.Adam(learning_rate=self.vf_lr)

        self.pi_loss_metric = tf.keras.metrics.Mean(name="pi_loss")
        self.v_loss_metric = tf.keras.metrics.Mean(name="V_loss")
        self.q_metric = tf.keras.metrics.Mean(name="Q")
        self.log_path = "./log/" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.summary_writer = tf.summary.create_file_writer(self.log_path + "/summary/")

    @tf.function
    def update_ppo(self, obs, act, adv, ret, logp):
        # self.actor_critic.train()
        # obs = tf.constant(obs)
        # act = tf.constant(act)
        # adv = tf.constant(adv)
        # ret = tf.constant(ret)
        logp_a_old = logp
        pi_loss, v_loss = 0., 0.

        for _ in tf.range(self.train_pi_iters):

            with tf.GradientTape() as tape:
                # pi, logp, logp_pi, mu
                _, logp_a, _, _ = self.actor_critic.policy(obs, act)
                ratio = tf.exp(logp_a - logp_a_old)  # pi(a|s) / pi_old(a|s)
                min_adv = tf.where(adv > 0, (1 + self.clip_ratio) * adv, (1 - self.clip_ratio) * adv)
                pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv, min_adv))

            gradients = tape.gradient(pi_loss, self.actor_critic.policy.trainable_weights)
            self.train_pi.apply_gradients(zip(gradients, self.actor_critic.policy.trainable_variables))

            kl = tf.reduce_mean(logp_a_old - logp_a)
            if kl > 1.5 * self.target_kl:
                break

        for _ in tf.range(self.train_v_iters):
            with tf.GradientTape() as tape:
                v = tf.squeeze(self.actor_critic.vf_mlp(obs))
                v_loss = tf.keras.losses.MSE(v, ret)

            gradients = tape.gradient(v_loss, self.actor_critic.vf_mlp.trainable_weights)
            self.train_v.apply_gradients(zip(gradients, self.actor_critic.vf_mlp.trainable_variables))

        self.pi_loss_metric.update_state(pi_loss)
        self.v_loss_metric.update_state(v_loss)

        return pi_loss, v_loss

    def train(self):
        start_time = time.time()
        o, r, d, ep_ret, ep_len, n_env_step = self.eval_env.reset(), 0, False, 0, 0, 0
        latests_100_score = deque(maxlen=100)

        for epoch in range(self.epochs):
            if (epoch == 0) or (((epoch + 1) % self.print_every) == 0):
                print("[%d/%d]" % (epoch + 1, self.epochs))
            o = self.env.reset()
            for t in range(self.steps_per_epoch):
                a, _, logp_t, v_t, _ = self.actor_critic(o.reshape(1, -1))

                o2, r, d, _ = self.env.step(a.numpy()[0])
                ep_ret += r
                ep_len += 1
                n_env_step += 1

                # Save the Experience to our buffer
                self.buf.store(o, a, r, v_t, logp_t)
                o = o2

                terminal = d or (ep_len == self.max_ep_len)
                if terminal or (t == (self.steps_per_epoch - 1)):
                    # if trajectory didn't reach terminal state, bootstrap value target
                    last_val = 0 if d else self.actor_critic.vf_mlp(tf.constant(o.reshape(1, -1))).numpy()[0][0]
                    self.buf.finish_path(last_val)
                    o, ep_ret, ep_len = self.env.reset(), 0, 0

            # Perform PPO update!
            obs, act, adv, ret, logp = [tf.constant(x) for x in self.buf.get()]
            self.update_ppo(obs, act, adv, ret, logp)

            # Evaluate
            if (epoch == 0) or (((epoch + 1) % self.evaluate_every) == 0):
                ram_percent = psutil.virtual_memory().percent  # memory usage
                print("[Eval. start] step:[%d/%d][%.1f%%] #step:[%.1e] time:[%s] ram:[%.1f%%]." %
                      (epoch + 1, self.epochs, epoch / self.epochs * 100,
                       n_env_step,
                       time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)),
                       ram_percent)
                      )
                o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
                _ = self.eval_env.render(mode='human')
                while not (d or (ep_len == self.max_ep_len)):
                    a, _, _, _ = self.actor_critic.policy(tf.constant(o.reshape(1, -1)))
                    o, r, d, _ = self.eval_env.step(a.numpy()[0])
                    _ = self.eval_env.render(mode='human')
                    ep_ret += r  # compute return
                    ep_len += 1
                print("[Evaluate] ep_ret:[%.4f] ep_len:[%d]" % (ep_ret, ep_len))
                latests_100_score.append((ep_ret))
                self.write_summary(epoch, latests_100_score, ep_ret, n_env_step)
                print("Saving weights...")
                self.actor_critic.save_weights(self.log_path + "/weights/weights")

    def write_summary(self, episode, latest_100_score, episode_score, total_step):

        with self.summary_writer.as_default():
            tf.summary.scalar("Reward (clipped)", episode_score, step=episode)
            tf.summary.scalar("Latest 100 avg reward (clipped)", np.mean(latest_100_score), step=episode)
            tf.summary.scalar("V_Loss", self.v_loss_metric.result(), step=episode)
            tf.summary.scalar("PI_Loss", self.pi_loss_metric.result(), step=episode)
            tf.summary.scalar("Total Frames", total_step, step=episode)

        self.v_loss_metric.reset_states()
        self.pi_loss_metric.reset_states()
        # self.q_metric.reset_states()

    def play(self, load_dir=None, trial=5):
        RENDER_ON_EVAL = True

        if load_dir:
            loaded_ckpt = tf.train.latest_checkpoint(load_dir)
            self.actor_critic.load_weights(loaded_ckpt)

        for i in range(trial):
            o, d, ep_ret, ep_len = self.eval_env.reset(), False, 0, 0
            if RENDER_ON_EVAL:
                _ = self.eval_env.render(mode='human')
            while not (d):
                a, _, _, _ = self.actor_critic.policy(tf.constant(o.reshape(1, -1)))
                o, r, d, _ = self.eval_env.step(a.numpy()[0])
                if RENDER_ON_EVAL:
                    _ = self.eval_env.render(mode='human')
                ep_ret += r  # compute return
                ep_len += 1
            print("[Evaluate] [%d/%d] ep_ret:[%.4f] ep_len:[%d]"
                  % (i, trial, ep_ret, ep_len))

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