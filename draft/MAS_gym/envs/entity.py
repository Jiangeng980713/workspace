import numpy as np
from enum import Enum
import sys


class Agent:
    def __init__(self, ID, infoMap=None, uncertaintyMap=None, targetMap=None, fromGlobalSize = None,shape=None, row=0, col=0,region_number=0):
        self.ID = ID
        self.row = row
        self.col = col
        self.SenseArea = np.asarray([  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                       [0, 0.0000, 0.0002, 0.0009, 0.0024, 0.0034, 0.0024, 0.0009, 0.0002, 0.0000, 0],
                                       [0, 0.0002, 0.0017, 0.0092, 0.0250, 0.0349, 0.0250, 0.0092, 0.0017, 0.0002, 0],
                                       [0, 0.0009, 0.0092, 0.0486, 0.1322, 0.1845, 0.1322, 0.0486, 0.0092, 0.0009, 0],
                                       [0, 0.0024, 0.0250, 0.1322, 0.3594, 0.5016, 0.3594, 0.1322, 0.0250, 0.0024, 0],
                                       [0, 0.0034, 0.0349, 0.1845, 0.5016, 0.7000, 0.5016, 0.1845, 0.0349, 0.0034, 0],
                                       [0, 0.0024, 0.0250, 0.1322, 0.3594, 0.5016, 0.3594, 0.1322, 0.0250, 0.0024, 0],
                                       [0, 0.0009, 0.0092, 0.0486, 0.1322, 0.1845, 0.1322, 0.0486, 0.0092, 0.0009, 0],
                                       [0, 0.0002, 0.0017, 0.0092, 0.0250, 0.0349, 0.0250, 0.0092, 0.0017, 0.0002, 0],
                                       [0, 0.0000, 0.0002, 0.0009, 0.0024, 0.0034, 0.0024, 0.0009, 0.0002, 0.0000, 0],
                                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

        self.individual_info = infoMap
        self.individual_uncertainty = uncertaintyMap
        self.detectSize = self.SenseArea.shape[0]
        self.fromGlobalSize = fromGlobalSize
        self.shape = shape
        self.cachedMap = []
        self.tempMap = np.ones((3, self.shape[0], self.shape[1]))
        self.tempMap[2] = 0
        self.PercentageInfo = 0.
        self.PercentageUncertainty = 0.
        self.communicate_activated = False
        self.last_comm = []
        self.comm = []
        self.target_status = None
        self.individual_targetMap = np.zeros(self.shape)
        self.target_temp = None
        self.region_number = region_number

    def setLocation(self, row, col):
        self.row = row
        self.col = col

    def getLocation(self):
        return [self.row, self.col]

    def move(self, action):
        """
        No movement: 0
        North (-1,0): 1
        East (0,1): 2
        South (1,0): 3
        West (0,-1): 4
        LeftUp (-1,-1) : 5
        RightUP (-1,1) :6
        RightDown (1,1) :7
        RightLeft (1,-1) :8
        check valid action of the agent. be sure not to be out of the boundary
        """
        if action == 0:
            return 0
        elif action == 1:
            self.row -= 1
        elif action == 2:
            self.col += 1
        elif action == 3:
            self.row += 1
        elif action == 4:
            self.col -= 1
        elif action == 5:
            self.row -= 1
            self.col -= 1
        elif action == 6:
            self.row -= 1
            self.col += 1
        elif action == 7:
            self.row += 1
            self.col += 1
        elif action == 8:
            self.row += 1
            self.col -= 1
        else:
            print("agent can only move NESW/1234")
            sys.exit()

    def reverseMove(self, action):
        if action == 0:
            return 0
        elif action == 1:
            self.row += 1
        elif action == 2:
            self.col -= 1
        elif action == 3:
            self.row -= 1
        elif action == 4:
            self.col += 1
        elif action == 5:
            self.row += 1
            self.col += 1
        elif action == 6:
            self.row += 1
            self.col -= 1
        elif action == 7:
            self.row -= 1
            self.col -= 1
        elif action == 8:
            self.row -= 1
            self.col += 1
        else:
            print("agent can only move NESW/1234")
            sys.exit()


    def updateInfo(self):
        min_row = max((self.row - self.detectSize // 2), 0)
        max_row = min((self.row + self.detectSize // 2 + 1), self.shape[0])
        min_col = max((self.col - self.detectSize // 2), 0)
        max_col = min((self.col + self.detectSize // 2 + 1), self.shape[1])

        transform_row = self.detectSize // 2 - self.row
        transform_col = self.detectSize // 2 - self.col

        infoMap_temp = np.copy(self.individual_info)
        unctertainty_temp = np.copy(self.individual_uncertainty)
        updateSense = (self.individual_info[min_row:max_row, min_col:max_col] + unctertainty_temp[min_row:max_row, min_col:max_col]) * \
                      (1-self.SenseArea[(min_row + transform_row):(max_row + transform_row), (min_col + transform_col):(max_col + transform_col)])
        self.individual_info[min_row:max_row, min_col:max_col] = np.minimum(infoMap_temp[min_row:max_row, min_col:max_col], updateSense)
        self.individual_info[self.individual_info < 0] = 0

    def updateUncertainty(self):
        min_row = max((self.row - self.detectSize // 2), 0)
        max_row = min((self.row + self.detectSize // 2 + 1), self.shape[0])
        min_col = max((self.col - self.detectSize // 2), 0)
        max_col = min((self.col + self.detectSize // 2 + 1), self.shape[1])

        transform_row = self.detectSize // 2 - self.row
        transform_col = self.detectSize // 2 - self.col

        self.individual_uncertainty[min_row:max_row, min_col:max_col] *= 1-self.SenseArea[(min_row + transform_row):(
                                                                                           max_row + transform_row),
                                                                                          (min_col + transform_col):(
                                                                                           max_col + transform_col)]
        self.individual_uncertainty[self.individual_uncertainty < 0] = 0

    def updatefromglobal(self, obs_info, obs_uncertainty):
        min_row = max((self.row - self.fromGlobalSize // 2), 0)
        max_row = min((self.row + self.fromGlobalSize // 2 + 1), self.shape[0])
        min_col = max((self.col - self.fromGlobalSize // 2), 0)
        max_col = min((self.col + self.fromGlobalSize // 2 + 1), self.shape[1])

        self.individual_info[min_row:max_row, min_col:max_col] = \
            obs_info[min_row:max_row, min_col:max_col]
        self.individual_uncertainty[min_row:max_row, min_col:max_col] =\
            obs_uncertainty[min_row:max_row, min_col:max_col]




class reward(Enum):
    MOVE        = -0.05
    MOVEDIAGONAL= -0.05*np.sqrt(2)
    NOMOVE      = -0.055
    COLLISION   = -0.20
    REMOVEDUST  = +0.20
    FINDTARGET  = +10.0
    COMMUNICATE = +1.0


class Target():
    def __init__(self, row, col, ID, time_found=np.nan):
        self.row = row
        self.col = col
        self.ID = ID
        self.time_found = time_found
        self.status = None

    def getLocation(self):
        return self.row, self.col

    def updateFound(self, timeStep):
        if np.isnan(self.time_found):
            self.time_found = timeStep
