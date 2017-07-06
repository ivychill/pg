# -*- coding: utf-8 -*-
#  train neural network
# 1. define network
# 2. define loss
# 3. train
# 4. evalute total reward

# TODO:
# 2 thread, one for play and produce samples, the other for consume samples and train network
# kafka is alternative solution

import os
import gym
import random
import numpy as np
import tensorflow as tf
import skimage
from skimage import transform, color, exposure
from collections import deque
import logging.handlers

# hyperparameters
EPISODE_NUM = 100000000
EPISODE_BATCH_SIZE = 1000
SAVE_INTERVAL = 1000000
GAMMA = 0.99 # discount factor for reward
# GAMMA = 1 # discount factor for reward
LEARNING_RATE = 1e-4 # feel free to play with this to train faster or more stably.
# ACTION_NUM = 18
ACTION_NUM = 9

CKPT_DIR = 'model'
LOG_DIR = 'log'
SUMMARY_DIR = 'summary/run070614'

MAX_LOG_SIZE = 2560000
LOG_BACKUP_NUM = 4000
logger = logging.getLogger('pg_tennis')
log_file = os.path.join(LOG_DIR, 'pg_tennis.log')
handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_NUM)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

tf.reset_default_graph()

feature_observation = tf.placeholder(tf.float32, [None, 84, 84, 4], name='feature_observation')
label_action = tf.placeholder(tf.float32, [None, ACTION_NUM], name='label_action')

W_conv1 = tf.get_variable("W_conv1", shape=[8, 8, 4, 32],
           initializer=tf.contrib.layers.xavier_initializer())
b_conv1 = tf.Variable(tf.zeros([32]))
h_conv1 = tf.nn.conv2d(feature_observation, W_conv1, strides=[1, 4, 4, 1], padding='SAME')
h_conv1_actv = tf.nn.relu(h_conv1 + b_conv1)

W_conv2 = tf.get_variable("W_conv2", shape=[4, 4, 32, 64],
           initializer=tf.contrib.layers.xavier_initializer())
b_conv2 = tf.Variable(tf.zeros([64]))
h_conv2 = tf.nn.conv2d(h_conv1_actv, W_conv2, strides=[1, 2, 2, 1], padding='SAME')
h_conv2_actv = tf.nn.relu(h_conv2 + b_conv2)

W_conv3 = tf.get_variable("W_conv3", shape=[3, 3, 64, 64],
           initializer=tf.contrib.layers.xavier_initializer())
b_conv3 = tf.Variable(tf.zeros([64]))
h_conv3 = tf.nn.conv2d(h_conv2_actv, W_conv3, strides=[1, 1, 1, 1], padding='SAME')
h_conv3_actv = tf.nn.relu(h_conv3 + b_conv3)

W_fc = tf.get_variable("W_fc", shape=[11 * 11 * 64, 512],
           initializer=tf.contrib.layers.xavier_initializer())
b_fc = tf.Variable(tf.zeros([512]))
h_conv3_actv_flat = tf.reshape(h_conv3_actv, [-1, 11 * 11 * 64])
h_fc = tf.matmul(h_conv3_actv_flat, W_fc) + b_fc
h_fc_actv = tf.nn.relu(h_fc)

W_output = tf.get_variable("W_output", shape=[512, ACTION_NUM],
           initializer=tf.contrib.layers.xavier_initializer())
b_output = tf.Variable(tf.zeros([ACTION_NUM]))
output = tf.matmul(h_fc_actv, W_output) + b_output
output_actv = tf.nn.softmax(output)

# our loss now looks like ∑Ailogp(yi∣xi), where yi is the action we happened to sample and Ai is a number that we call an advantage.
# in multiclass policy gradient, lable = ∑ (advantage * log(softmax()))
advantage = tf.placeholder(tf.float32, name="reward_signal")
log_likelihood = tf.reduce_sum(label_action * tf.log(tf.clip_by_value(output_actv,1e-10,1.0)), 1)
loss = -tf.reduce_mean(log_likelihood * advantage)
tf.summary.histogram('log_likelihood', log_likelihood)
tf.summary.scalar('loss', loss)

# train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE) # Our optimizer
trained_var = tf.trainable_variables()
new_grad = tf.gradients(loss, trained_var)
W_conv1_grad = tf.placeholder(tf.float32, name="W_conv_grad1")
b_conv1_grad = tf.placeholder(tf.float32, name="b_conv_grad1")
W_conv2_grad = tf.placeholder(tf.float32, name="W_conv_grad2")
b_conv2_grad = tf.placeholder(tf.float32, name="b_conv_grad2")
W_conv3_grad = tf.placeholder(tf.float32, name="W_conv_grad3")
b_conv3_grad = tf.placeholder(tf.float32, name="b_conv_grad3")
W_fc_grad = tf.placeholder(tf.float32, name="W_fc_grad")
b_fc_grad = tf.placeholder(tf.float32, name="b_fc_grad")
W_output_grad = tf.placeholder(tf.float32, name="W_output_grad")
b_output_grad = tf.placeholder(tf.float32, name="b_output_grad")
batch_grad = [W_conv1_grad, b_conv1_grad, W_conv2_grad, b_conv2_grad, W_conv3_grad, b_conv3_grad, W_fc_grad, b_fc_grad, W_output_grad, b_output_grad]
update_grad = adam.apply_gradients(zip(batch_grad, trained_var))

# preprocess the screen image
def preprocess(processed_input):
    processed_output = processed_input[30:250, 8:160]  # crop
    processed_output = skimage.color.rgb2gray(processed_output)
    processed_output = skimage.transform.resize(processed_output,(84,84))
    processed_output = skimage.exposure.rescale_intensity(processed_output,out_range=(0,255))
    return processed_output # 84*84

# action space
# PLAYER_A_NOOP:
# PLAYER_A_FIRE:
# PLAYER_A_UP:
# PLAYER_A_RIGHT:
# PLAYER_A_LEFT:
# PLAYER_A_DOWN:
# PLAYER_A_UPRIGHT:
# PLAYER_A_UPLEFT:
# PLAYER_A_DOWNRIGHT:
# PLAYER_A_DOWNLEFT:
# PLAYER_A_UPFIRE:
# PLAYER_A_RIGHTFIRE:
# PLAYER_A_LEFTFIRE:
# PLAYER_A_DOWNFIRE:
# PLAYER_A_UPRIGHTFIRE:
# PLAYER_A_UPLEFTFIRE:
# PLAYER_A_DOWNRIGHTFIRE:
# PLAYER_A_DOWNLEFTFIRE:
# 9 actions without 'fire' is omitted deliberately because of env bug.
def map_action_from_label_to_env(from_act):
    to_act = from_act
    if from_act == 0:
        to_act = 1
    elif (from_act >= 1 and from_act < 9):
        to_act = from_act + 9
    return to_act

def sample(distribution):
    act = np.random.choice(ACTION_NUM, p=distribution)
    env_act = map_action_from_label_to_env(act)
    return act, env_act

# input: action array
# def one_hot_encode_label(actions):
#     label_code = np.zeros((actions.size, ACTION_NUM))
#     label_code[np.arange(actions.size), actions] = 1
#     return label_code
def one_hot_encode_label(action):
    label_code = np.zeros(ACTION_NUM)
    label_code[action] = 1
    return label_code

# return RL(Reinforce Learning) return, i.e. PG(Policy Gradient) advantage
def discount_rewards(rewards_):
    discounted_r = []
    running_add = 0
    reversed_rewards = rewards_
    reversed_rewards.reverse()
    for _, _reward in enumerate(reversed_rewards):
        running_add = running_add * GAMMA + _reward
        discounted_r.append(running_add)
    discounted_r.reverse()
    # print "discounted_r: ", discounted_r
    return discounted_r

episode_i = 0
done = False
reward = 0.0
observations, actions, rewards = [],[],[]
stacked_obs = np.zeros([1,84,84,4])
replay_buffer = deque()
reward_sum = 0

env = gym.make("Tennis-v0")
observation = env.reset()

# Merge all the summaries and write them out to /tmp/mnist_logs (by default)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

# Launch the graph
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.log_device_placement = True
with tf.Session(config = config) as sess:
    sess.run(init)
    file_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)

    # Reset the gradient placeholder. We will collect gradients in grad_buffer until we are ready to update our policy network.
    grad_buffer = sess.run(trained_var)
    for index, grad in enumerate(grad_buffer):
        grad_buffer[index] = grad * 0

    saver = tf.train.Saver()

    while episode_i < EPISODE_NUM:
        # tennis unit: set, game, point
        # change default episode from one set to one point
        while (reward == 0.0):
            # Run the policy network and get an action to take.
            # env.render()
            processed_obs = np.reshape(preprocess(observation), [1, 84, 84, 1])
            stacked_obs = np.append(stacked_obs[:, :, :, 1:], processed_obs, axis=3)
            observations.append(stacked_obs)
            action_distribution = sess.run(output_actv, feed_dict={feature_observation: stacked_obs}) # predict
            action, env_action = sample(np.reshape(action_distribution, [ACTION_NUM]))  # action_distribution if of [1, ACTION_NUM]
            one_hot_action = one_hot_encode_label(action)
            actions.append(one_hot_action)
            observation, reward, done, info = env.step(env_action)
            rewards.append(reward)
            # logger.debug('action: %d, %d, reward: %d, done: %r' % (action, env_action, reward, done))
            reward_sum += reward

        returns = discount_rewards(rewards)

        minibatch_feature = np.vstack(observations)
        # logger.debug('minibatch_feature: %s' % (minibatch_feature))
        minibatch_label = np.vstack(actions)
        minibatch_return = np.vstack(returns)

        matrix_grad, summary, cur_loglik, cur_loss = sess.run([new_grad, merged, log_likelihood, loss], feed_dict={feature_observation: minibatch_feature,
                                                    label_action: minibatch_label,
                                                    advantage: minibatch_return})

        for index, grad in enumerate(matrix_grad):
            grad_buffer[index] += grad
            # logger.debug("grad_buffer[%d]: %s" % (index, grad_buffer))

        if episode_i % EPISODE_BATCH_SIZE == (EPISODE_BATCH_SIZE - 1):
            sess.run(update_grad, feed_dict={W_conv1_grad: grad_buffer[0], b_conv1_grad: grad_buffer[1],
                                             W_conv2_grad: grad_buffer[2], b_conv2_grad: grad_buffer[3],
                                             W_conv3_grad: grad_buffer[4], b_conv3_grad: grad_buffer[5],
                                             W_fc_grad: grad_buffer[6], b_fc_grad: grad_buffer[7],
                                             W_output_grad: grad_buffer[8], b_output_grad: grad_buffer[9]})

            for index, grad in enumerate(grad_buffer):
                grad_buffer[index] = grad * 0

            file_writer.add_summary(summary, episode_i)
            logger.warn('episode %d, loglik %s, loss %s, average reward %f' % (episode_i, cur_loglik, cur_loss, reward_sum / EPISODE_BATCH_SIZE))
            if reward_sum > 0:
                logger.warn('Task solved in %d episodes!' % (episode_i))
            reward_sum = 0

        if episode_i % SAVE_INTERVAL == (SAVE_INTERVAL - 1):
            logger.warn('save model at epoch %d' % (episode_i))
            ckpt_file = os.path.join(CKPT_DIR, 'pg_tennis_model')
            saver.save(sess, ckpt_file, (episode_i + 1) / SAVE_INTERVAL)

        episode_i += 1
        reward = 0.0
        stacked_obs = np.zeros([1, 84, 84, 4])
        if done:
            done = False
            observation = env.reset()
        observations, actions, rewards = [], [], []  # reset array memory

env.close()