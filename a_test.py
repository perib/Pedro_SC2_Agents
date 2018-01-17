import numpy as np
import tensorflow as tf
from pr_neuralnets import *


def main():
    nets = genSimpleFC2(4, 1)
    nets = TrainQLearning(nets, 1, 0.0001)
    init_op = init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    for i in range(0, 100000):
        if i % 1000 == 0:
            print(i)

        input = np.random.rand(4)

        target = [[input.sum()]]  # + self.gamma * pred_reward
        train, pred_reward, loss,targettf = sess.run(
            [nets['train_step'],nets['predicted_reward'],nets["loss"],nets["target"] ],
            feed_dict={nets['input_to_net']: [input],
                       nets['target']: target, nets['keep_prob']: 1})

        if i % 100 == 0:
            print("things")
            print("pred ", pred_reward)
            print("target ",target)
            print(loss)
            print("actual loss" , (pred_reward-target)**2  )

    for i in range(0, 10):
        input = np.random.rand(4)
        out = sess.run(
            [nets['predicted_reward']],
            feed_dict={nets['input_to_net']: [input],
                       nets['keep_prob']: 1})
        print(input)
        print(out[0])



main()