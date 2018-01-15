import tensorflow as tf
import numpy as np


def genSimpleFC(intput_length,output_length):
    nets = {}
    with tf.name_scope('input'):
        nets['input_to_net'] = tf.placeholder(tf.float32, [None, intput_length], name='input_to_net')

    with tf.name_scope('fc_1'):
        W_fc1 = weight_variable([intput_length, 512])
        b_fc1 = bias_variable([512])
        nets['fc_1'] = tf.nn.relu(tf.matmul(nets['input_to_net'], W_fc1) + b_fc1)

        nets['keep_prob'] = tf.placeholder(tf.float32)
        nets['h_fc1_drop'] = tf.nn.dropout(nets['fc_1'], nets['keep_prob'])

    with tf.name_scope('fc_2'):
        W_fc2 = weight_variable([512, output_length])
        b_fc2 = bias_variable([output_length])
        nets['predicted_reward'] = tf.matmul(nets['h_fc1_drop'], W_fc2) + b_fc2

    return nets

def TrainQLearning(nets,output_length,learning_rate):
    with tf.variable_scope("loss"):
        nets["target"] = tf.placeholder(tf.float32, [None, output_length], name='target')
        nets["loss"] = tf.losses.mean_squared_error(labels=nets["target"],predictions=nets['predicted_reward'])
        nets['loss_summary'] = tf.summary.scalar("loss_summary", nets["loss"])


    with tf.variable_scope("TrainStep"):
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-08
        nets['global_step'] = tf.Variable(0, name='global_step', trainable=False)
        nets['train_step'] = tf.train.AdamOptimizer(learning_rate).minimize(nets["loss"],global_step=nets['global_step'])


    with tf.name_scope("Reward_Summary"):
        nets['Reward'] = tf.placeholder(tf.float32, (), name='Reward')
        nets['Rewardsum'] = tf.summary.scalar("Rewardsum", nets['Reward'])

    with tf.name_scope("Count_Tracker"):
        nets['current_epoch'] = tf.Variable(0, name='current_epoch', trainable=False, dtype=tf.int32)
        nets['increment_current_epoch'] = tf.assign(nets['current_epoch'], nets['current_epoch'] + 1)

        nets['current_step'] = tf.Variable(0, name='current_step', trainable=False, dtype=tf.int32)
        nets['increment_current_step'] = tf.assign(nets['current_step'], nets['current_step'] + 1)
        nets['reset_current_step'] = tf.assign(nets['current_step'], 0)

    return nets

def weight_variable(shape, name=None, freeze_weight = None):
    if freeze_weight == None:
        initial = tf.truncated_normal(shape, stddev=0.001)
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(freeze_weight[name], name=name, trainable = False)

def bias_variable(shape, name=None,freeze_bias = None):
    if freeze_bias == None:
        initial = tf.constant(0.0, shape=shape)
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(freeze_bias[name], name=name,trainable = False)