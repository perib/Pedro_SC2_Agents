import numpy as np
import tensorflow as tf
from pr_neuralnets import *


def main():
    nets = genSimpleFC2(4, 1)
    nets = TrainQLearning(nets, 1, 0.1)
    init_op = init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    for i in range(0, 10000):
        if i % 1000 == 0:
            print(i)

        input = np.random.rand(4)

        target = [[input[0]]]  # + self.gamma * pred_reward
        train = sess.run(
            [nets['train_step']],
            feed_dict={nets['input_to_net']: [input],
                       nets['target']: target, nets['keep_prob']: 1})

    for i in range(0, 10):
        input = np.random.rand(4)
        out = sess.run(
            [nets['predicted_reward']],
            feed_dict={nets['input_to_net']: [input],
                       nets['keep_prob']: 1})

        print(out[0])


main()