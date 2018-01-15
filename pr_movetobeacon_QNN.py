import numpy as np
#https://gamescapad.es/building-bots-in-starcraft-2-for-psychologists/#installation
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import random
import time
import dill
import tensorflow

from pr_neuralnets import *

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_PLAYER_FRIENDLY = 1
_PLAYER_NEUTRAL = 3  # beacon/minerals
_PLAYER_HOSTILE = 4
_NO_OP = actions.FUNCTIONS.no_op.id
_MOVE_SCREEN = actions.FUNCTIONS.Move_screen.id
_ATTACK_SCREEN = actions.FUNCTIONS.Attack_screen.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_NOT_QUEUED = [0]
_SELECT_ALL = [0]

_SELECT_POINT = actions.FUNCTIONS.select_point.id


_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_MARINE = 48


#Commands

_N = 0
_NE = 1
_E = 2
_SE = 3
_S = 4
_SW = 5
_W = 6
_NW = 7
_Stay = 8

Commands = [_N,_NE,_E,_SE,_E,_S,_SW,_W,_NW]#_Stay]

class MovetoBeaconQ(base_agent.BaseAgent):
    """An agent specifically for solving the MoveToBeacon map."""
    def __init__(self):
        super(MovetoBeaconQ, self).__init__()
        self.selected =[0,0]
        self.gamma = 0.999
        self.nets = genSimpleFC(intput_length=5, output_length=1)
        self.nets = TrainQLearning(self.nets, output_length=1, learning_rate=0.1)

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

        self.prev_state = None
        self.prev_action = None


        self.startE = 1  # Starting chance of random action
        self.endE = 0.01 #.1  # Final chance of random action
        self.anneling_steps = 1500  # How many steps of training to reduce startE to endE.
        self.pre_train_steps = 10000  # How many steps of random actions before training begins.
        self.stepDrop = (self.startE - self.endE) / self.anneling_steps #updated every episode
        self.preStepcount = 0  # keeps track of steps needed before training
        self.e = self.startE

        self.prev_marine_x = None
        self.prev_marine_y = None

        self.episode_step = 0

    def prediction(self,marine_x,marine_y,beacon_x,beacon_y,action):
        pred_reward = self.sess.run(
            [self.nets['predicted_reward']],
            feed_dict={self.nets['input_to_net']: [[marine_x, marine_y, beacon_x, beacon_y, action]],
                       self.nets['keep_prob']: 1})
        return action, pred_reward[0][0][0]

    def predict_action(self,marine_x,marine_y,beacon_x,beacon_y):
        bestaction = 0
        bestreward = float('-inf')
        for action in Commands:
            #print(action)

            pred_reward = self.sess.run(
                [self.nets['predicted_reward']],
                feed_dict={self.nets['input_to_net']: [[marine_x,marine_y,beacon_x,beacon_y,action]],self.nets['keep_prob']: 1})

            #
            # print(pred_reward)

            #print(pred_reward)
            if pred_reward[0][0][0] > bestreward:
                bestreward = pred_reward[0][0][0]
                bestaction = action
                #self.prev_state = [[marine_x, marine_y, beacon_x, beacon_y, action]]


        return bestaction, bestreward



    def get_action(self,action,marine_x,marine_y):
        if marine_x == None or marine_y == None:
            return actions.FunctionCall(_NO_OP, [])
        stepsize = 3
        if action == _N:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x+stepsize,marine_y] ])
        if action == _NE:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x+stepsize,marine_y+stepsize] ])
        if action == _E:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x,marine_y+stepsize] ])
        if action == _SE:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x-stepsize,marine_y+stepsize] ])
        if action == _S:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x-stepsize,marine_y] ])
        if action == _SW:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x-stepsize,marine_y-stepsize] ])
        if action == _W:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x,marine_y-stepsize] ])
        if action == _NW:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x+stepsize,marine_y-stepsize] ])
        if action == _Stay:
            return actions.FunctionCall(_NO_OP, [])

        print("nooope")
        return actions.FunctionCall(_NO_OP, [])

    def train_network(self,reward,current_state,pred_reward):
        if current_state == 'terminal':
            target = [[reward]] #+ self.gamma * pred_reward
            #print(type(target))
            train = self.sess.run(
                [self.nets['train_step']],
                feed_dict={self.nets['input_to_net']: self.prev_state,
                           self.nets['target']: target, self.nets['keep_prob']: 1})
        else:
            target = [[reward + self.gamma * pred_reward]]
            #print(type(target))
            train = self.sess.run(
                [self.nets['train_step']],
                feed_dict={self.nets['input_to_net']: self.prev_state,
                           self.nets['target']: target, self.nets['keep_prob']: 1})

    def reset(self):
        super(MovetoBeaconQ, self).reset()
        self.episode_step = 0


    def step(self, obs):
        super(MovetoBeaconQ, self).step(obs)

        if self.episode_step == 0:
            print("total episodes %d, won %d, percent %f" %(self.episodes,self.reward, (self.reward/self.episodes) ))

        self.episode_step += 1
        #print(features)
        #print(features.SCREEN_FEATURES)

        #time.sleep(0.1)



        if obs.last():
            #self.train_network(obs.reward, "terminal", 0)

            if self.e > self.endE and self.preStepcount >= self.pre_train_steps:
                self.e = self.e - self.stepDrop
            self.prev_state = None
            self.prev_action = None
            return actions.FunctionCall(_NO_OP, [])

        # learn from previous step
        #if obs.last() and self.prev_action != None:
        #    reward = obs.reward
        #    self.train_network(reward, 'terminal', None)
        #    self.prev_state = None
        #    self.prev_action = None
        #    return actions.FunctionCall(_NO_OP, [])

        #if marine not selected - select marine
        if not _MOVE_SCREEN in obs.observation["available_actions"]:
            unit_type = obs.observation['screen'][_UNIT_TYPE]
            unit_y, unit_x = (unit_type == _TERRAN_MARINE).nonzero()
            if unit_y.any():
                i = random.randint(0, len(unit_y) - 1)
                target = [unit_x[i], unit_y[i]]
                self.selected = target

            #print("oh fuck")
            #print(obs.observation["available_actions"])
            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
        else:

            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if not neutral_y.any():
                return actions.FunctionCall(_NO_OP, [])

            friendly_y, friendly_x = (player_relative == _PLAYER_FRIENDLY).nonzero()
            if len(friendly_x) == 0 or len(friendly_y) == 0:
                marine_x = self.prev_marine_x #if in a beacon you cant see your thing
                marine_y = self.prev_marine_y
            else:
                marine_x = friendly_x.mean()
                marine_y = friendly_y.mean()

            beacon_x = neutral_x.mean()
            beacon_y = neutral_y.mean()

            #get action
            if np.random.rand(1) < self.e or self.preStepcount < self.pre_train_steps:
                action = np.random.choice(Commands,1)[0]
                pred_action, pred_reward = self.prediction(marine_x, marine_y, beacon_x, beacon_y, action)
                self.preStepcount = self.preStepcount + 1
            else:
                pred_action, pred_reward = self.predict_action(marine_x,marine_y,beacon_x,beacon_y)



            if self.prev_action != None:
                if obs.reward > 0:
                    self.train_network(obs.reward, "terminal", 0)
                else:
                    self.train_network(0, "not terminal", pred_reward)

                #self.train_network(0, "not terminal", pred_reward)
                #if not obs.last():
                #    self.train_network(0, "not terminal", pred_reward)
                #else:
                #    self.train_network(0, "not terminal", pred_reward)

            self.prev_action = pred_action
            self.prev_state = [[marine_x,marine_y,beacon_x,beacon_y,pred_action]]

            #print(pred_action)
            return self.get_action(pred_action, marine_x,marine_y)



            #Random actions!
            #return self.get_action(random.randint(0,9),marine_x,marine_y)

