# -*- coding: utf-8 -*-
#  train neural network
# 1. define network
# 2. define loss
# 3. train
# 4. evalute total reward


import os
import gym
import numpy as np
import tensorflow as tf
import skimage
from skimage import color
# from skimage import transform, color, exposure
from collections import deque
import logging.handlers

# hyperparameters
EPISODE_NUM = 5000000
PURE_EXPLORE_NUM = 100000
INITIAL_EPSILON = 0.5 # starting value of epsilon
EPSILON_DECAY_RATE = 1-1e-5 # final value of epsilon
BATCH_SIZE = 1 # every how many episodes to do a param update?
SAVE_INTERVAL = 10000
GAMMA = 0.99 # discount factor for reward
LEARNING_RATE = 1e-4 # feel free to play with this to train faster or more stably.
# ACTION_NUM = 18
ACTION_NUM = 9
# PIC_HEIGHT = 250
# PIC_WIDTH = 160
# PIC_CHANNEL = 3
PIC_HEIGHT = 220
PIC_WIDTH = 152
PIC_CHANNEL = 1
CONV_LAYER_WIDTH = 32
FC_LAYER_WIDTH = 512

REPLAY_BUFFER_SIZE = 50000 # number of previous transitions to remember
OBSERV_BATCH_SIZE = 3200. # timesteps to observe before training
SGD_BATCH_SIZE = 32 # size of minibatch

CKPT_DIR = 'model'
LOG_DIR = 'log'
SUMMARY_DIR = 'summary/run062110'

MAX_LOG_SIZE = 2560000
LOG_BACKUP_NUM = 4000
logger = logging.getLogger('pg_tennis')
log_file = os.path.join(LOG_DIR, 'pg_tennis.log')
handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=MAX_LOG_SIZE, backupCount=LOG_BACKUP_NUM)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)

# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)

# def variable_summaries(var):
#   """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
#   with tf.name_scope('summaries'):
#     mean = tf.reduce_mean(var)
#     tf.summary.scalar('mean', mean)
#     with tf.name_scope('stddev'):
#       stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
#     tf.summary.scalar('stddev', stddev)
#     tf.summary.scalar('max', tf.reduce_max(var))
#     tf.summary.scalar('min', tf.reduce_min(var))
#     tf.summary.histogram('histogram', var)

tf.reset_default_graph()

# observation_space Box(250, 160, 3)
# action_space Discrete(18)
feature_observation = tf.placeholder(tf.float32, [None, PIC_HEIGHT, PIC_WIDTH, PIC_CHANNEL], name='feature_observation')
label_action = tf.placeholder(tf.float32, [None, ACTION_NUM], name='label_action')
# tf.summary.image('input_image', feature_observation)

# convolution layer, ReLU activation, max pooling
# W_conv = tf.Variable(tf.zeros([5, 5, 3, CONV_LAYER_WIDTH]))
# W_conv = tf.Variable(tf.zeros([8, 8, 1, CONV_LAYER_WIDTH]))
# W_conv = weight_variable([8, 8, 1, CONV_LAYER_WIDTH])
W_conv = tf.get_variable("W_conv", shape=[8, 8, 1, CONV_LAYER_WIDTH],
           initializer=tf.contrib.layers.xavier_initializer())
b_conv = tf.Variable(tf.zeros([CONV_LAYER_WIDTH]))
# with tf.name_scope('W_conv'):
#     variable_summaries(W_conv)
# with tf.name_scope('b_conv'):
#     variable_summaries(b_conv)

h_conv = tf.nn.conv2d(feature_observation, W_conv, strides=[1, 4, 4, 1], padding='SAME')
h_conv_actv = tf.nn.relu(h_conv + b_conv)
h_pool = tf.nn.max_pool(h_conv_actv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# tf.summary.histogram('h_conv', h_conv)
# tf.summary.histogram('h_conv_actv', h_conv_actv)
# tf.summary.histogram('h_pool', h_pool)

# fully collected layer.
# W_fc = tf.Variable(tf.zeros([28 * 19 * CONV_LAYER_WIDTH, FC_LAYER_WIDTH]))
# W_fc = weight_variable([28 * 19 * CONV_LAYER_WIDTH, FC_LAYER_WIDTH])
W_fc = tf.get_variable("W_fc", shape=[28 * 19 * CONV_LAYER_WIDTH, FC_LAYER_WIDTH],
           initializer=tf.contrib.layers.xavier_initializer())
b_fc = tf.Variable(tf.zeros([FC_LAYER_WIDTH]))
# with tf.name_scope('W_fc'):
#     variable_summaries(W_fc)
# with tf.name_scope('b_fc'):
#     variable_summaries(b_fc)

h_pool_flat = tf.reshape(h_pool, [-1, 28 * 19 * CONV_LAYER_WIDTH])
h_fc = tf.matmul(h_pool_flat, W_fc) + b_fc
h_fc_actv = tf.nn.relu(h_fc)
# tf.summary.histogram('h_pool_flat', h_pool_flat)
# tf.summary.histogram('h_fc', h_fc)
# tf.summary.histogram('h_fc_actv', h_fc_actv)

# W_output = tf.Variable(tf.zeros([FC_LAYER_WIDTH, ACTION_NUM]))
# W_output = weight_variable([FC_LAYER_WIDTH, ACTION_NUM])
W_output = tf.get_variable("W_output", shape=[FC_LAYER_WIDTH, ACTION_NUM],
           initializer=tf.contrib.layers.xavier_initializer())
b_output = tf.Variable(tf.zeros([ACTION_NUM]))
# with tf.name_scope('W_output'):
#     variable_summaries(W_output)
# with tf.name_scope('b_output'):
#     variable_summaries(b_output)

output = tf.matmul(h_fc_actv, W_output) + b_output
output_actv = tf.nn.softmax(output)
# tf.summary.histogram('output', output)/settings/keys
# tf.summary.histogram('output_actv', output_actv)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_action * tf.log(output_actv), reduction_indices=[1]))
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_action, logits=output))

# our loss now looks like ∑Ailogp(yi∣xi), where yi is the action we happened to sample and Ai is a number that we call an advantage.
# in multiclass policy gradient, lable = ∑ (advantage * log(softmax()))
trained_var = tf.trainable_variables()
advantage = tf.placeholder(tf.float32, name="reward_signal")
# log_likelihood = tf.reduce_sum(label_action * tf.log(output_actv), reduction_indices=[1])
log_likelihood = tf.reduce_sum(label_action * tf.log(tf.clip_by_value(output_actv,1e-10,1.0)), 1)
loss = -tf.reduce_mean(log_likelihood * advantage)
new_grad = tf.gradients(loss, trained_var)
tf.summary.histogram('log_likelihood', log_likelihood)
tf.summary.scalar('loss', loss)

# Once we have collected a series of gradients from multiple episodes, we apply them.
# We don't just apply gradeients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE) # Our optimizer
W_conv_grad = tf.placeholder(tf.float32, name="W_conv_grad")
b_conv_grad = tf.placeholder(tf.float32, name="b_conv_grad")
W_fc_grad = tf.placeholder(tf.float32, name="W_fc_grad")
b_fc_grad = tf.placeholder(tf.float32, name="b_fc_grad")
W_output_grad = tf.placeholder(tf.float32, name="W_output_grad")
b_output_grad = tf.placeholder(tf.float32, name="b_output_grad")
batch_grad = [W_conv_grad, b_conv_grad, W_fc_grad, b_fc_grad, W_output_grad, b_output_grad]
update_grad = adam.apply_gradients(zip(batch_grad, trained_var))


# preprocess the screen image
def preprocess(processed_input):
    processed_output = processed_input[30:250, 8:160]  # crop
    processed_output = skimage.color.rgb2gray(processed_output)
    processed_output = np.reshape(processed_output, [PIC_HEIGHT, PIC_WIDTH, PIC_CHANNEL])
    return processed_output

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

# def map_action_from_env_to_label(from_act):
#     to_act = from_act
#     if from_act == 1:
#         to_act = 0
#     elif (from_act >= 10 and from_act < 18):
#         to_act = from_act - 9
#     return to_act

# distribution is a numpy ndarray
# two choice: 1.sample according to distribution; 2.epsilon-greedy
def sample(distribution, episode_no):
    # act = np.random.choice(ACTION_NUM, p=distribution)
    act = np.random.randint(0, ACTION_NUM)
    if episode_no >= PURE_EXPLORE_NUM:
        epsilon = INITIAL_EPSILON * (EPSILON_DECAY_RATE ** episode_no)
        rand = np.random.random()
        if rand > epsilon:
            optimal_act = np.argmax(distribution)
            # convert type from numpy.int64 to native python int
            act = np.asscalar(optimal_act)

    env_act = map_action_from_label_to_env(act)
    return act, env_act

# input: action array
def one_hot_encode_label(actions):
    label_code = np.zeros((actions.size, ACTION_NUM))
    label_code[np.arange(actions.size), actions] = 1
    return label_code

# return RL(Reinforce Learning) return, i.e. PG(Policy Gradient) advantage
def discount_rewards(arr_reward):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(arr_reward)
    running_add = 0
    for index in reversed(range(arr_reward.size)):
        running_add = running_add * GAMMA + arr_reward[index]
        discounted_r[index] = running_add
    return discounted_r

episode_i = 0
done = False
reward = 0.0
observations, actions, rewards = [],[],[]
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

    # Reset the gradient placeholder. We will collect gradients in
    # gradBuffer until we are ready to update our policy network.
    grad_buffer = sess.run(trained_var)
    for index, grad in enumerate(grad_buffer):
        grad_buffer[index] = grad * 0

    saver = tf.train.Saver()

    while episode_i < EPISODE_NUM:
        # tennis unit: set, game, point
        # change default episode from one set to one point
        # while not done:
        while (reward == 0.0):
            # Run the policy network and get an action to take.
            env.render()
            processed_obs = np.reshape(preprocess(observation), [1, PIC_HEIGHT, PIC_WIDTH, PIC_CHANNEL])
            observations.append(processed_obs)
            action_distribution = sess.run(output_actv, feed_dict={feature_observation: processed_obs})
            action, env_action = sample(np.reshape(action_distribution, [ACTION_NUM]), episode_i)  # action_distribution if of [1, ACTION_NUM]
            actions.append(action)
            observation, reward, done, info = env.step(env_action)
            # print 'action: ', action, '|env_action: ', env_action, '|reward: ', reward, '|done: ', done
            logger.debug('action: %d, %d, reward: %d, done: %r' % (action, env_action, reward, done))
            rewards.append(reward)
            reward_sum += reward

        matrix_feature = np.vstack(observations)
        # matrix_label = np.vstack(one_hot_encode_label(np.asarray(actions))) # because of one-hot, vector become matrix
        matrix_label = one_hot_encode_label(np.asarray(actions)) # because of one-hot, vector become matrix
        array_reward = np.vstack(rewards)

        # compute the discounted reward backwards through time
        vector_return = discount_rewards(array_reward)
        # size the rewards to be unit normal (helps control the gradient estimator variance)
        # vector_return -= np.mean(vector_return)
        # vector_return /= np.std(vector_return)

        # Get the gradient for this episode, and save it in the gradBuffer
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata = tf.RunMetadata()
        # matrix_grad, summary = sess.run([new_grad, merged],
        #                                 feed_dict={feature_observation: matrix_feature,
        #                                             label_action: matrix_label,
        #                                             advantage: vector_return},
        #                                 options=run_options,
        #                                 run_metadata=run_metadata)

        matrix_grad, summary = sess.run([new_grad, merged], feed_dict={feature_observation: matrix_feature,
                                                    label_action: matrix_label,
                                                    advantage: vector_return})

        file_writer.add_summary(summary, episode_i)

        for index, grad in enumerate(matrix_grad):
            grad_buffer[index] += grad
            logger.debug("grad_buffer[%d]: %s"
                        % (index, grad_buffer))

        curr_W_conv, curr_b_conv, curr_W_fc, curr_b_fc, curr_W_output, curr_b_output, curr_loglik, curr_loss \
            = sess.run([W_conv, b_conv, W_fc, b_fc, W_output, b_output, log_likelihood, loss],
                                             {feature_observation: matrix_feature,
                                              label_action: matrix_label,
                                              advantage: vector_return})
        logger.debug("W_conv: %s, b_conv: %s, W_fc: %s, b_fc: %s, W_output: %s, b_output:%s, loglik: %s, loss: %s"
              % (curr_W_conv, curr_b_conv, curr_W_fc, curr_b_fc, curr_W_output, curr_b_output, curr_loglik, curr_loss))

        # If we have completed enough episodes, then update the policy network with our gradients.
        if episode_i % BATCH_SIZE == (BATCH_SIZE - 1):
            sess.run(update_grad, feed_dict={W_conv_grad: grad_buffer[0], b_conv_grad: grad_buffer[1],
                                             W_fc_grad: grad_buffer[2], b_fc_grad: grad_buffer[3],
                                             W_output_grad: grad_buffer[4], b_output_grad: grad_buffer[5]})

            for index, grad in enumerate(grad_buffer):
                grad_buffer[index] = grad * 0

            # Give a summary of how well our network is doing for each batch of episodes.
            # running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
            logger.warn('Average reward for episode %d : %f.' % (episode_i, reward_sum / BATCH_SIZE))

            if reward_sum / BATCH_SIZE > 0:
                logger.warn('Task solved in %d episodes!' % (episode_i))
                # break

            reward_sum = 0

        if episode_i % SAVE_INTERVAL == (SAVE_INTERVAL - 1):
            # Now, save the graph
            ckpt_file = os.path.join(CKPT_DIR, 'pg_tennis_model')
            saver.save(sess, ckpt_file, episode_i)

        episode_i += 1
        # done = False
        # observation = env.reset()
        reward = 0.0
        if done:
            done = False
            observation = env.reset()
        observations, actions, rewards = [], [], []  # reset array memory

env.close()