import numpy as np
#https://gamescapad.es/building-bots-in-starcraft-2-for-psychologists/#installation
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
import random
import time
import dill

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

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_TERRAN_MARINE = 48

class CollecteMinerals(base_agent.BaseAgent):
  """An agent specifically for solving the MoveToBeacon map."""

  def __init__(self):
      super(CollecteMinerals, self).__init__()
      self.Select = True
      self.selected =[0,0]

  def step(self, obs):
    super(CollecteMinerals, self).step(obs)

    with open('obs.pkl','wb') as f:
        dill.dump(obs,f)

    #print(features)
    #print(features.SCREEN_FEATURES)

    time.sleep(0.1)

    if self.Select:
        self.Select = False
        unit_type = obs.observation['screen'][_UNIT_TYPE]
        unit_y, unit_x = (unit_type == _TERRAN_MARINE).nonzero()
        if unit_y.any():
            print(len(unit_y))
            i = random.randint(0, len(unit_y) - 1)
            target = [unit_x[i], unit_y[i]]
            self.selected = target

            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
    else:
        self.Select = True
        if _MOVE_SCREEN in obs.observation["available_actions"]:
            player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
            neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
            if not neutral_y.any():
                return actions.FunctionCall(_NO_OP, [])

            closest, min_dist = None, None
            for p in zip(neutral_x, neutral_y):
                dist = np.linalg.norm(np.array(self.selected) - np.array(p))
                if not min_dist or dist < min_dist:
                    closest, min_dist = p, dist

            return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, closest])

    '''if _MOVE_SCREEN in obs.observation["available_actions"]:
      player_relative = obs.observation["screen"][_PLAYER_RELATIVE]
      neutral_y, neutral_x = (player_relative == _PLAYER_NEUTRAL).nonzero()
      if not neutral_y.any():
        return actions.FunctionCall(_NO_OP, [])
      target = [int(neutral_x.mean()), int(neutral_y.mean())]
      return actions.FunctionCall(_MOVE_SCREEN, [_NOT_QUEUED, target])
    else:
      return actions.FunctionCall(_SELECT_ARMY, [_SELECT_ALL])'''