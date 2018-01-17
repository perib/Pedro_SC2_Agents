import numpy as np
#https://gamescapad.es/building-bots-in-starcraft-2-for-psychologists/#installation
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import random
import time
import dill
import tensorflow
import math
import copy
import tensorflow as tf
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

_Cheat = 9

#Commands = [_N,_NE,_SE,_E,_S,_SW,_W,_NW,_Cheat]#_Stay]

#Commands = [_N,_E,_S,_W,_Cheat]

#Commands = [_N,_E,_S,_W]

Commands = [_N,_NE,_SE,_E,_S,_SW,_W,_NW]

# [Observations1, action, reward, observations2, terminal]
#stores the new memories and replaces old ones
class experience_buffer():

    def __init__(self, buffer_size = 1000):
        self.buffer = [] #stores objects in an arrray of length buffer_size
        self.buffer_size = buffer_size #max size of array
        self.current = 0 #current index to be replaced

    def add(self, experience):
        if len(self.buffer)  >= self.buffer_size: #if the buffer is full
            if(self.current == self.buffer_size-1): #if we are at the end of the list, replace the first item
                self.current = 0
            self.buffer[self.current] = experience #replace item
            self.current += 1 #set to replace the following number
        else:
            self.buffer.append(experience) #if not full, just add new item

    def sample(self, size): #randomly sample size number of instances
        return random.sample(self.buffer, size)


class MovetoBeaconQ(base_agent.BaseAgent):
    """An agent specifically for solving the MoveToBeacon map."""
    def __init__(self):
        super(MovetoBeaconQ, self).__init__()
        self.selected =[0,0]
        self.gamma = 0.9
        self.nets = genSimpleFC2(intput_length=5, output_length=1)
        self.nets = TrainQLearning(self.nets, output_length=1, learning_rate=0.1)

        init_op = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init_op)

        self.prev_state = None
        self.prev_action = None


        self.startE = 1  # Starting chance of random action
        self.endE = 0.01 #.1  # Final chance of random action
        self.anneling_steps = 500  # How many steps of training to reduce startE to endE.
        self.pre_train_steps = 750  # How many steps of random actions before training begins.
        self.stepDrop = (self.startE - self.endE) / self.anneling_steps #updated every episode
        self.preStepcount = 0  # keeps track of steps needed before training
        self.e = self.startE

        self.prev_marine_x = None
        self.prev_marine_y = None

        self.episode_step = 0

        self.prev_distance = float('inf')


        self.batchsize = 40
        self.buffersize = 5000

        self.update_freq = 250

        self.buffer = experience_buffer(self.buffersize)


        tensorboard_folder = "./tbfolder"
        merged = tf.summary.merge_all()
        tb_writer = tf.summary.FileWriter(tensorboard_folder, self.sess.graph)

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



    def get_action(self,action,marine_x,marine_y,beacon_x, beacon_y):
        if marine_x == None or marine_y == None:
            return actions.FunctionCall(_NO_OP, [])
        stepsize = 5

        if stepsize + marine_x > 83:
            up = 83
        else:
            up = stepsize + marine_x

        if stepsize + marine_y > 83:
            right = 83
        else:
            right = stepsize + marine_y

        if -stepsize + marine_y < 0:
            left = 0
        else:
            left = -stepsize + marine_y

        if -stepsize + marine_x < 0:
            down = 0
        else:
            down = -stepsize + marine_x

        if action == _N:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [up,marine_y] ])
        if action == _NE:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [up,right] ])
        if action == _E:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x,right] ])
        if action == _SE:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [down,right] ])
        if action == _S:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [down,marine_y] ])
        if action == _SW:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [down,left] ])
        if action == _W:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [marine_x,left] ])
        if action == _NW:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [up,left] ])
        if action == _Stay:
            return actions.FunctionCall(_NO_OP, [])

        if action == _Cheat:
            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, [beacon_x, beacon_y]])

        return actions.FunctionCall(_NO_OP, [])

    #self.buffer.add([copy.deepcopy(self.prev_state), copy.deepcopy(self.prev_action), obs.reward + 100, "terminal", 0])

    def train_on_batch(self):
        memories = self.buffer.sample(self.batchsize)

        for memory in memories:
            prevstate = memory[0]
            prevaction = memory[1]
            reward = memory[2]
            current_state = memory[3]
            predreward = memory[4]

            prevstate_action = prevstate


            if current_state == 'terminal':
                target = [[reward]] #+ self.gamma * pred_reward
                train = self.sess.run(
                    [self.nets['train_step']],
                    feed_dict={self.nets['input_to_net']: prevstate_action,
                               self.nets['target']: target, self.nets['keep_prob']: 1})
            else:
                target = [[reward + self.gamma * predreward]]
                #print(type(target))
                train,loss = self.sess.run(
                    [self.nets['train_step'],self.nets['loss']],
                    feed_dict={self.nets['input_to_net']: prevstate_action,
                               self.nets['target']: target, self.nets['keep_prob']: 1})

                #print(loss)

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

        if (self.steps % self.update_freq == 0 and len(self.buffer.buffer) > self.batchsize):
            self.train_on_batch()


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
                self.prev_marine_x = marine_x
                self.prev_marine_y = marine_y


            beacon_x = neutral_x.mean()
            beacon_y = neutral_y.mean()

            dist = math.sqrt((beacon_x-marine_x)**2 + (beacon_y-marine_y)**2)

            #get action
            if np.random.rand(1) < self.e or self.preStepcount < self.pre_train_steps:
                action = np.random.choice(Commands,1)[0]
                pred_action, pred_reward = self.prediction(marine_x, marine_y, beacon_x, beacon_y, action)
                self.preStepcount = self.preStepcount + 1
            else:

                pred_action, pred_reward = self.predict_action(marine_x,marine_y,beacon_x,beacon_y)

            pred_action = int(pred_action)

            if self.prev_action != None:
                if obs.reward > 0:

                    self.buffer.add([copy.deepcopy(self.prev_state),copy.deepcopy(self.prev_action),obs.reward+1000,"terminal",0])

                  #  self.train_network(obs.reward+100, "terminal", 0)
                else:
                    if self.prev_distance > dist:
                        reward  = 100
                    else:
                        reward = -100
                    self.prev_distance = dist

                    self.buffer.add(
                        [copy.deepcopy(self.prev_state), copy.deepcopy(self.prev_action), reward, "not terminal",
                         pred_reward])

                  #  self.train_network(reward, "not terminal", pred_reward)

                #self.train_network(0, "not terminal", pred_reward)
                #if not obs.last():
                #    self.train_network(0, "not terminal", pred_reward)
                #else:
                #    self.train_network(0, "not terminal", pred_reward)

            self.prev_action = pred_action
            self.prev_state = [[marine_x,marine_y,beacon_x,beacon_y,pred_action]]

            #print(pred_action)
            return self.get_action(pred_action, marine_x,marine_y,beacon_x, beacon_y)



            #Random actions!
            #return self.get_action(random.randint(0,9),marine_x,marine_y)

