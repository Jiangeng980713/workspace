import gym
import time
import math
import sys
import numpy as np
from gym.envs.classic_control import rendering
from MAS_gym.envs.draw import *
from MAS_gym.envs.entity import *
from Decomposed import Decompose_and_Search
from Decomposed import Region
import matplotlib.pyplot as pyplot

# show all number in the matrix
# np.set_printoptions(threshold=10000)

# threading
from threading import Lock

# OPPOSITE_ACTIONS = {1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
# agentColor = (1, 0.2, 0.6)
# agentCommColor = (1, 0.6, 0.2)
# obstacleColor = (0., 0., 0.)
# targetNotFound = (0., 1., 0.)
# targetFound = (0.545, 0.27, 0.075)
# highestProbColor = (1., 0., 0.)
# highestUncertaintyColor = (0., 0., 1.)
# lowestProbColor = (1., 1., 1.)
# boundary color
# boundarycolor = (0., 0., 0.)

REGION = np.load('region_map.npy')
AGENT = np.load('agent_map.npy')
INFO = np.load('info.npy')
LINE = np.load('line.npy')
TARGET = np.load('target.npy')
# TARGET = np.load('target.npy')

class MAS_gym(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, state0=None, numAgents=8, rangeNumTargets=(10, 16), obs_density=0., removeSpeed0=0.001,
                 shapes=(20, 64), observationSize=11, rangeDistribs=(16, 32), model="B",
                 REGION = None, AGENT = None, BOUNDARY = None , INFO = None, TARGET = None):
        """
        NOTES:
        """
        super(MAS_gym, self).__init__()  # Define action and observation space

        # They must be gym.spaces objects    # Example when using discrete actions:
        self.viewer = None
        self.worldState = None
        self.infoMap = None
        self.uncertaintyMap = None
        self.targetMap = None
        self.agents = []
        self.targets = []
        self.rangeNumTargets = rangeNumTargets
        self.numAgents = numAgents
        self.rewards = np.zeros((self.numAgents,))
        self.shapes = shapes
        self.shape = None
        self.observationSize = observationSize
        self.fromGlobalSize = None
        self.obs_density = obs_density
        self.removeSpeed0 = removeSpeed0
        self.removeSpeed = None  # variable speed to guarantee the reward from the dust is the same
        self.varyWorldSize()
        self.maxInfo = 0.
        self.minInfo = 0.
        self.maxUncertainty = 0.
        self.minUncertainty = 0.
        self.totalInfo = 0.
        self.totalUncertainty = 0.
        self.scalar = 0.
        self.rangeDistribs = rangeDistribs
        self.communicateCircle = 15
        self.communicateTimes = None
        self.newCommunicateTimes = None
        self.model = model

        # decomposed region map###
        self.region_map = REGION
        self.agent_location_map = AGENT # 0 empty 1 agent
        self.boundary_line = BOUNDARY   # 0 line 1 empty
        self.info_input = INFO
        self.target_input = TARGET

        self.setWorld(state0)
        self.finished = False
        self.agent_mode = [1] * self.numAgents
        self.level = [0] * self.numAgents
        self.action_map = [None for _ in range(self.numAgents)]
        self.gapsize = [np.random.randint(1, 6) for _ in range(self.numAgents)]

        # threading
        self.mutex = Lock()

        # some parameters
        self.OPPOSITE_ACTIONS = {1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}
        self.agentColor = (1, 0.2, 0.6)
        self.agentCommColor = (1, 0.6, 0.2)
        self.obstacleColor = (0., 0., 0.)
        self.targetNotFound = (0., 1., 0.)
        self.targetFound = (0.545, 0.27, 0.075)
        self.highestProbColor = (1., 0., 0.)
        self.highestUncertaintyColor = (0., 0., 1.)
        self.lowestProbColor = (1., 1., 1.)
        # boundary color
        self.boundarycolor = (0., 0., 0.)

    def varyWorldSize(self):
        shapes = np.random.randint(self.shapes[0], self.shapes[1] + 1)
        self.shape = (shapes, shapes)

    def addGaussian(self, variance):
        """
        1. generate some random matrix represent the mean and variance of the distribution
        2. create the distribution in a 20x20 matrix
        3. for now the distribution contains decimal and the sum of the whole matrix of each distribution is 1
        4. the position of the distribution is the mean in a list (row,col)
        """

        gaussian_mean = self.shape[0] * np.random.rand(1, 2)[0]
        gaussian_var = np.zeros((2, 2))
        # gaussian_var[([0, 1], [0, 1])] = 0.2 * self.shape[0] * np.random.rand(1, 2)[0]
        # TODO: add the suitable value of the variance
        gaussian_var[([0, 1], [0, 1])] = variance * np.random.rand(1, 2)[0]
        row_mat, col_mat = np.meshgrid(np.linspace(0, self.shape[0] - 1, self.shape[0]),
                                       np.linspace(0, self.shape[1] - 1, self.shape[1]))
        SigmaX = np.sqrt(gaussian_var[0][0])
        SigmaY = np.sqrt(gaussian_var[1][1])
        Covariance = gaussian_var[0][1]
        r = Covariance / (SigmaX * SigmaY)
        coefficients = 1 / (2 * math.pi * SigmaX * SigmaY * np.sqrt(1 - math.pow(r, 2)))
        p1 = -1 / (2 * (1 - math.pow(r, 2)))
        px = np.power(row_mat - gaussian_mean[0], 2) / gaussian_var[0][0]
        py = np.power(col_mat - gaussian_mean[1], 2) / gaussian_var[1][1]
        pxy = 2 * r * (row_mat - gaussian_mean[0]) * (col_mat - gaussian_mean[1]) / (SigmaX * SigmaY)
        distribution_matrix = coefficients * np.exp(p1 * (px - pxy + py))
        return distribution_matrix

    def createProbability(self):
        """
        1. using the addGaussian function to generate some distribution
        2. the number of the distribution is random.
        3. sum the all these distribution matrix
        4. all the elements in the matrix is divided by the the the numDistriution to get the normalized map 
        then multiple (1 - probability out of the distribution)
        5. uniform distribute the probability out of the distribution in the 20x20 matrix and we get the whole map of 
        sum up to 1        
        6. increase the number in the probability matrix
        enlarge ratio is to increase the minimum number in the matrix to 1
        7. all the elements in the matrix is multiple by the enlarge ratio and usn the function round()
        8. create the infoMap
        """

        numDistribs = np.random.randint(self.rangeDistribs[0], self.rangeDistribs[1] + 1)
        infoMaps = np.array([self.addGaussian(10) for _ in range(numDistribs)])

        # define probability out of the distribution by the baseline
        # baseline = 0.1  # baseline of "empty" regions is 1-baseline = 0.1
        # infoMap = (1 - baseline) * np.mean(infoMaps, axis=0) + baseline * np.ones(self.shape) / (
        #         self.shape[0] * self.shape[1])
        infoMap = np.mean(infoMaps, axis=0)
        infoMap /= np.sum(infoMap)
        maxInfo = np.max(infoMap)
        # scalar = 0.5/maxInfo * np.random.rand() + 0.5/maxInfo
        infoMap = infoMap/maxInfo
        return infoMap

    def createUncertainty(self):
        """
        create the uncertainty of the map, and this map is used to set the targets
        """
        numDistribs = np.random.randint(self.rangeDistribs[0], self.rangeDistribs[1] + 1)
        uncertaintyMap = np.array([self.addGaussian(20) for _ in range(numDistribs)])
        uncertaintyMap = np.mean(uncertaintyMap, axis=0)
        uncertaintyMap /= np.sum(uncertaintyMap)
        uncertaintyMap /= np.max(uncertaintyMap)

        return uncertaintyMap


    def setTarget(self, infoMap, uncertaintyMap, manulTaget=False):
        """
        1. if the manualTarget is 1, set the target manually --> DON'T DO IT FOR NOW (TODO)
        2. if the manualTarget is 0, using the probability matrix
        3. sum all the element in the probability
        4. get the vector of sum probability and choose the position of the target
        5. all the target in the matrix is set to be -2, initially. Which means hidden. And if they are found,
        change it to -3
        """

        targets = []
        # infoMap = infoMap/self.scalar
        infoMap = infoMap / np.max(infoMap)
        uncertaintyMap = uncertaintyMap / np.max(uncertaintyMap)
        target_prob = infoMap + (2 * np.random.rand() - 1) * uncertaintyMap
        target_prob = np.clip(target_prob, 0, 1)
        target_prob = target_prob/np.sum(target_prob)
        numTargets = np.random.randint(self.rangeNumTargets[0], self.rangeNumTargets[1] + 1)
        flat_im = [item for sublist in list(map(list, target_prob)) for item in sublist]
        # check the uncertaintyMap to set the target
        target_positions = []
        targetMap = np.zeros(self.shape)
        num_target = 0

        while num_target < numTargets:
            randomPos = np.random.choice(len(flat_im), p=flat_im)
            # target_positions = [(r // self.shape[0], r % self.shape[0]) for r in randomPos]
            target_position = (randomPos // self.shape[0], randomPos % self.shape[0])
            if np.random.rand() < (1 - uncertaintyMap[target_position[0]][target_position[1]]):
                # Avoid overlapping of agent and target ###
                if self.agent_location_map[target_position[0]][target_position[1]] == 0:
                    target_positions.append(target_position)
                    num_target += 1

        for tg in range(len(target_positions)):
            tx, ty = target_positions[tg]
            if targetMap[tx][ty] != 0:
                continue # target already there
            targets.append(Target(ID=tg + 1, row=tx, col=ty))
            targetMap[tx][ty] = -2

        return targetMap, targets

    def setRandomTargets(self):
        targets = []
        # numTargets = len(self.targets)
        numTargets = np.random.randint(self.rangeNumTargets[0], self.rangeNumTargets[1] + 1)
        target_positions = []
        targetMap = np.zeros(self.shape)
        num_target = 0
        while num_target < numTargets:
            target_position = (np.random.randint(self.shape[0]), np.random.randint(self.shape[1]))
            target_positions.append(target_position)
            num_target += 1

        for tg in range(len(target_positions)):
            tx, ty = target_positions[tg]
            if targetMap[tx][ty] != 0:
                continue  # target already there
            targets.append(Target(ID=tg + 1, row=tx, col=ty))
            targetMap[tx][ty] = -2

        return targetMap, targets

    def setInverseTargets(self, infoMap, uncertaintyMap):
        targets = []
        infoMap = infoMap / np.sum(infoMap)
        inverseInfo = np.max(infoMap) - infoMap
        inverseInfo = inverseInfo/np.sum(inverseInfo)
        # numTargets = len(self.targets)
        numTargets = np.random.randint(self.rangeNumTargets[0], self.rangeNumTargets[1] + 1)
        flat_im = [item for sublist in list(map(list, inverseInfo)) for item in sublist]
        # check the uncertaintyMap to set the target
        target_positions = []
        targetMap = np.zeros(self.shape)
        num_target = 0

        while num_target < numTargets:
            randomPos = np.random.choice(len(flat_im), p=flat_im)
            # target_positions = [(r // self.shape[0], r % self.shape[0]) for r in randomPos]
            target_position = (randomPos // self.shape[0], randomPos % self.shape[0])
            # if np.random.rand() < (1 - uncertaintyMap[target_position[0]][target_position[1]]):
            target_positions.append(target_position)
            num_target += 1

        for tg in range(len(target_positions)):
            tx, ty = target_positions[tg]
            if targetMap[tx][ty] != 0:
                continue  # target already there
            targets.append(Target(ID=tg + 1, row=tx, col=ty))
            targetMap[tx][ty] = -2

        return targetMap, targets
    # read the map and load the target with locations

    def readTarget(self):
        target_temp=self.target_input
        targets, n_targets = [], 1
        targetMap = np.zeros(self.shape)
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if target_temp[row][col] == -2 or target_temp[row][col] == -3:
                    targets.append(Target(ID=n_targets, row=row, col=col,
                                          time_found=(0. if target_temp[row][col] == -3 else np.nan)))
                    n_targets = n_targets + 1
                    targetMap[row][col] = target_temp[row][col]

        return targetMap, targets



    def setAgentandObstacle(self, targetMap, info_Map, uncertainty_Map):
        """
        Agents:
        1. add self.numAgents agents to the environment at random positions
        2. the position of the agents is set to be 1 to N (the ID of the agent)
        3. the uncertainty map is a local one, only available for individual agent except for communication
        Obstacles:
        1. generate some random position in a world matrix (check not the same as the target or other agents)
        2. all the obstacle is set to be -1
        Then add this to the worldState
        """
        if self.model == "A":
            self.fromGlobalSize = 2 * self.shape[0]
        elif self.model == "B":
            self.fromGlobalSize = 9
        elif self.model == "C":
            self.fromGlobalSize = 9
        worldState = np.zeros(self.shape)
        if self.obs_density > 0:
        # TODO: either make sure we get exactly the right obstacle density, OR maybe add obstacle chunks (lakes/hills)
            worldState[np.logical_and(
                np.random.rand(self.shape[0], self.shape[1]) < self.obs_density, targetMap == 0)] = -1
            # agent has been there can not set the obstacles ###
            worldState[np.logical_and(worldState, self.agent_location_map == 1)] = 0

        # extract agent location ###
        temp_row = []
        temp_col = []
        for i in range(0, self.agent_location_map.shape[0]):
            for j in range(0, self.agent_location_map.shape[1]):
                if self.agent_location_map[i][j] == 1:
                    temp_row.append(i)
                    temp_col.append(j)

        # initial agent location and region_number
        agents = []
        for a in range(1, self.numAgents + 1):
            row = temp_row[a - 1]
            col = temp_col[a - 1]
            worldState[row][col] = a

            # get the region number for each agent ###
            agent_region_number = self.region_map[row][col]

            # build a select matrix ###
            select_matrix = np.where(self.region_map != agent_region_number, 0, 1)

            # select info and uncertainty map, different regions unseen ###
            info_Map_temper = info_Map
            info_Map_temper = info_Map_temper * select_matrix

            uncertainty_Map_temper = uncertainty_Map
            uncertainty_Map_temper = uncertainty_Map_temper * select_matrix

            agent = Agent(ID=a, row=row, col=col, infoMap=np.copy(info_Map_temper), uncertaintyMap=np.copy(uncertainty_Map_temper),
                          fromGlobalSize=self.fromGlobalSize, shape=self.shape, region_number=agent_region_number)
            agents.append(agent)

        return worldState, agents

    def setWorld(self, state0=None, targetMode=0):
        """
        1. empty all the element
        2. create the new episode
        """
        # read maps and produce two maps###

        if state0 is not None:
            assert ((len(state0) == 4 and len(state0[0].shape) == 2
                     and len(state0[1].shape) == 2) or len(state0.shape) == 3)
            assert (state0[0].shape == state0[1].shape)
            # assert (np.asarray([np.sum(state0[2][state0[2] == a]) == a for a in range(1, self.numAgents + 1)]).all())
            # assert (np.sum(state0[2] > self.numAgents) == 0)

            self.shape = state0[0].shape

            targets, n_targets = [], 1
            targetMap = np.zeros(self.shape)
            for row in range(self.shape[0]):
                for col in range(self.shape[1]):
                    if state0[3][row][col] == -2 or state0[3][row][col] == -3:
                        targets.append(Target(ID=n_targets, row=row, col=col,
                                              time_found=(0. if state0[3][row][col] == -3 else np.nan)))
                        n_targets = n_targets + 1
                        targetMap[row][col] = state0[3][row][col]
            self.targetMap = targetMap
            self.targets = targets

            # create the world manually
            worldState = state0[2]
            worldState[worldState == -2] = 0
            worldState[worldState == -3] = 0
            self.worldState = worldState
            self.infoMap = state0[0]
            self.uncertaintyMap = state0[1]
            # maxInfo = np.max(self.infoMap)
            # self.scalar = 0.5 / maxInfo * np.random.rand() + 0.5 / maxInfo
            self.scalar = np.sum(self.infoMap)
            # self.infoMap = self.infoMap * self.scalar
            # self.uncertaintyMap = state0[1] / np.max(state0[1])
            if self.model == "A":
                self.fromGlobalSize = 2 * self.shape[0]
            elif self.model == "B":
                self.fromGlobalSize = 9
            elif self.model == "C":
                self.fromGlobalSize = 9

            if np.sum(state0[2] > 0) <= 0:
                self.worldState, self.agents = self.setAgentandObstacle(self.targetMap, self.infoMap, self.uncertaintyMap)
                self.rewards = np.zeros((self.numAgents,))
            else:
                agents = []
                self.numAgents = np.sum(state0[2] > 0)
                for a in range(1, self.numAgents + 1):
                    abs_pos = np.argmax(state0[2] == a)
                    row, col = abs_pos // self.shape[0], abs_pos % self.shape[0]
                    agents.append(Agent(ID=a, row=row, col=col, infoMap=np.copy(self.infoMap),
                                        uncertaintyMap=np.copy(self.uncertaintyMap), fromGlobalSize=self.fromGlobalSize
                                        , shape=self.shape))
                self.agents = agents

            self.rewards = np.zeros((self.numAgents,))
            for t in self.targets:
                t.status = np.array([False for _ in range(len(self.agents))])
            for a in self.agents:
                a.target_status = np.array([self.targets[i].status for i in range(len(self.targets))])
            self.minInfo, self.maxInfo = np.min(self.infoMap), np.max(self.infoMap)
            self.minUncertainty, self.maxUncertainty = np.min(self.uncertaintyMap), np.max(self.uncertaintyMap)
            self.totalUncertainty = np.sum(self.uncertaintyMap)
            self.totalInfo = np.sum(self.infoMap)
            self.communicateTimes = 0
            self.newCommunicateTimes = 0

        # create the world automatically
        # state0 = None
        else:
            self.varyWorldSize()
            self.infoMap = self.info_input
            self.uncertaintyMap = self.createUncertainty()
            # if targetMode == 0:
            #     self.targetMap, self.targets = self.setTarget(self.infoMap, self.uncertaintyMap)
            # elif targetMode == 1:
            #     self.targetMap, self.targets = self.setRandomTargets()
            # elif targetMode == 2:
            #     self.targetMap, self.targets = self.setInverseTargets(self.infoMap, self.uncertaintyMap)

            # read the exist targetMap
            self.targetMap, self.targets = self.readTarget()

            self.worldState, self.agents = self.setAgentandObstacle(self.targetMap, self.infoMap, self.uncertaintyMap)
            for t in self.targets:
                t.status = np.array([False for _ in range(len(self.agents))])
            for a in self.agents:
                a.target_status = np.array([self.targets[i].status for i in range(len(self.targets))])
            self.minInfo, self.maxInfo = np.min(self.infoMap), np.max(self.infoMap)
            self.minUncertainty, self.maxUncertainty = np.min(self.uncertaintyMap), np.max(self.uncertaintyMap)
            self.totalUncertainty = np.sum(self.uncertaintyMap)
            self.totalInfo = np.sum(self.infoMap)
            self.communicateTimes = 0
            self.newCommunicateTimes = 0

    #   now we get 3 channels to represent the world
    #   1. probability self.infoMap
    #   2. world map self.worldState containing empty--0 obstacles(-1) agents(1 to N) and target(-2 hidden or -3 found)
    #   This is the initial state of the world

    def communicate(self, agentID):
        """
        the agent will exchange the individual maps and change their own maps to the the minimum uncertainty
        requirement: the agents is close enough---the distance is smaller than 3*3 square
        method: the uncertainty map is stored in the agent but the agent can only have partial observation, through the
        communication they can get the map from other agent
        """
        # TODO: think about the communication, how to use the communication, enlarge the observation size of
        #       the uncertainty map?
        # TODO: add the communication in synchronize()?

        min_row = max((self.agents[agentID - 1].row - self.communicateCircle // 2), 0)
        max_row = min((self.agents[agentID - 1].row + self.communicateCircle // 2 + 1), self.shape[0])
        min_col = max((self.agents[agentID - 1].col - self.communicateCircle // 2), 0)
        max_col = min((self.agents[agentID - 1].col + self.communicateCircle // 2 + 1), self.shape[1])

        cachedUnc, cachedInfo, cachedTstatus = [], [], []

        for row in range(min_row, max_row):
            for col in range(min_col, max_col):
                if self.worldState[row][col] > 0 and self.worldState[row][col] != self.agents[agentID - 1].ID and \
                        (row - self.agents[agentID - 1].row) ** 2 + (col - self.agents[agentID - 1].col) ** 2 <= (
                        self.communicateCircle // 2) ** 2:
                    ID = int(self.worldState[row][col])
                    # judge agents' region numbers are same or not###
                    if self.agents[ID - 1].region_number == self.agents[agentID - 1].region_number:
                        self.agents[agentID - 1].communicate_activated = True
                        self.communicateTimes += 1  # should be divided by 2 because the communication is two-way.
                        self.agents[agentID - 1].comm.append(self.worldState[row][col])
                        cachedUnc.append(self.agents[ID - 1].individual_uncertainty)
                        cachedInfo.append(self.agents[ID - 1].individual_info)
                        cachedTstatus.append(self.agents[ID - 1].target_status)

        m = set(self.agents[agentID - 1].comm)
        n = set(self.agents[agentID - 1].last_comm)
        self.newCommunicateTimes += len(m) - len(m & n)
        self.agents[agentID - 1].last_comm = self.agents[agentID - 1].comm
        self.agents[agentID - 1].comm = []
        if cachedInfo:
            self.agents[agentID - 1].tempMap[1] = np.min(np.array(cachedInfo), axis=0)
        if cachedUnc:
            self.agents[agentID - 1].tempMap[0] = np.min(np.array(cachedUnc), axis=0)
        if cachedTstatus != []:
            cachedTstatus = np.sum(np.array(cachedTstatus), axis=0)
            self.agents[agentID - 1].target_temp = cachedTstatus > 0
            cachedTstatus = np.sum(cachedTstatus, axis=1)
            cachedTstatus = cachedTstatus > 0
            for i in range(len(cachedTstatus)):
                if cachedTstatus[i]:
                    self.agents[agentID - 1].tempMap[2][self.targets[i].row][self.targets[i].col] = -3


    def updateMap(self, agentID):
        """
        update the agent individual maps
        """
        amount_uncertain = np.sum(self.agents[agentID - 1].individual_uncertainty -
                                  np.min((self.agents[agentID - 1].individual_uncertainty,
                                          self.agents[agentID - 1].tempMap[0]), axis=0)) / self.totalUncertainty
        amount_info = np.sum(self.agents[agentID - 1].individual_info -
                             np.min((self.agents[agentID - 1].individual_info,
                                     self.agents[agentID - 1].tempMap[1]), axis=0)) / self.totalInfo
        self.agents[agentID - 1].individual_uncertainty = np.min((self.agents[agentID - 1].individual_uncertainty,
                                                                  self.agents[agentID - 1].tempMap[0]), axis=0)

        self.agents[agentID - 1].individual_info = np.min((self.agents[agentID - 1].individual_info,
                                                           self.agents[agentID - 1].tempMap[1]), axis=0)

        self.agents[agentID - 1].individual_targetMap = np.min((self.agents[agentID - 1].individual_targetMap,
                                                                self.agents[agentID - 1].tempMap[2]), axis=0)
        r = (amount_info + amount_uncertain) * reward.COMMUNICATE.value * 0.2
        self.rewards[agentID - 1] += r

    #   get the state of the agents and information map and update the map
    def extractObservation(self, agent):
        """
        1. input the agent (class)
        2. create the list of matrix to represent the observation layer of the agent every layer
        in the matrix is a 11*11 matrix
        the [0]*11*11 is the infoMap matrix, [1]*11*11 is the position of the other agent
        [2]*11*11 is the obstacle matrix, the [3]*11*11 is the sense area
        ********
        The sense area is fixed. It represents the area in which the target can be found
        We use a sense area with a gaussian distribution to represent the probability to find a target
        Sense area is a matrix smaller than the 11*11 like n*n and the n<11
        we use the 5*5 matrix, the area outside the n*n is set to be zero
        the array is defined in Agent.SenseArea
        ********
        3. get the obstacles and the other agents position around the agent as [0]*11*11
        4. get the dust map around the agent as [1]*11*11
        5. make a judgement
        if the dust in zero, the observation can extend to the target matrix and assign sense area of
        the target matrix to the [2]*11*11
        if the dust is non-zero the [2]*11*11 is set to be the default 0 matrix
        """

        transform_row = self.observationSize // 2 - agent.row
        transform_col = self.observationSize // 2 - agent.col

        # extreme location###
        min_row = max((agent.row - self.observationSize // 2), 0)
        max_row = min((agent.row + self.observationSize // 2 + 1), self.shape[0])
        min_col = max((agent.col - self.observationSize // 2), 0)
        max_col = min((agent.col + self.observationSize // 2 + 1), self.shape[1])

        observation = np.full((self.observationSize, self.observationSize), -1)
        targetMap = np.full((self.observationSize, self.observationSize), 0)
        infoMap = np.full((self.observationSize, self.observationSize), 0.)
        uncertaintyMap = np.full((self.observationSize, self.observationSize), 0.)

        # build a observe filter matrix###
        filter = np.full((max_row - min_row, max_col - min_col),0)

        for i in range(min_row, max_row):
            for j in range(min_col, max_col):
                if self.region_map[i][j] == agent.region_number:
                    filter[i - min_row][j - min_col] = 1
                else:
                    filter[i - min_row][j - min_col] = 0

        observation[(min_row + transform_row):(max_row + transform_row),
                    (min_col + transform_col):(max_col + transform_col)] = self.worldState[
                                                                           min_row:max_row, min_col:max_col]
        observation[(min_row + transform_row):(max_row + transform_row),
                    (min_col + transform_col):(max_col + transform_col)] = filter*observation[(min_row + transform_row):(max_row + transform_row),
                                                                                                (min_col + transform_col):(max_col + transform_col)]

        infoMap[(min_row + transform_row):(max_row + transform_row),
                (min_col + transform_col):(max_col + transform_col)] = agent.individual_info[
                                                                       min_row:max_row, min_col:max_col]


        targetMap[(min_row + transform_row):(max_row + transform_row),
                  (min_col + transform_col):(max_col + transform_col)] = filter * agent.individual_targetMap[
                                                                         min_row:max_row, min_col:max_col]

        uncertaintyMap[(min_row + transform_row):(max_row + transform_row),
                       (min_col + transform_col):(max_col + transform_col)] = agent.individual_uncertainty[
                                                                              min_row:max_row, min_col:max_col]

        observation_layers = np.zeros((5, self.observationSize, self.observationSize))
        # nearby agents (including me)
        observation_layers[0] = observation > 0
        # nearby obstacles
        observation_layers[1] = np.logical_or(observation == -1, observation > 0)
        # nearby information distribution
        observation_layers[2] = infoMap
        # nearby uncertainty distribution
        observation_layers[3] = uncertaintyMap
        # nearby found targets
        observation_layers[4] = targetMap == -3

        return observation_layers

    def listNextValidActions(self, agent_id, prev_action=0):
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
        available_actions = [0]

        # Get current agent
        agent = self.agents[agent_id - 1]  # did we offset or pad?

        MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)]  # off by one index
        for action in range(8):
            out_of_bounds = agent.row + MOVES[action][0] >= self.shape[0] \
                            or agent.row + MOVES[action][0] < 0 \
                            or agent.col + MOVES[action][1] >= self.shape[1] \
                            or agent.col + MOVES[action][1] < 0

            if (not out_of_bounds) \
                    and self.worldState[agent.row + MOVES[action][0], agent.col + MOVES[action][1]] == 0 \
                    and not (prev_action == self.OPPOSITE_ACTIONS[action + 1]):
                available_actions.append(action + 1)

        return available_actions

    def executeAction(self, agent, action, timeStep):
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
        """
        origLoc = agent.getLocation()
        # Move N,E,S,W
        if (action >= 1) and (action <= 8):
            agent.move(action)
            row, col = agent.getLocation()

            # If the move is not valid, roll it back, if the agent goes other region roll back
            if ((row < 0) or (col < 0) or (row >= self.shape[0]) or (col >= self.shape[1]) or
                (self.worldState[row][col] > 0) or (self.worldState[row][col] == -1)) or \
                    agent.region_number != self.region_map[row][col]:
                agent.reverseMove(action)
                self.rewards[agent.ID - 1] = reward.COLLISION.value
                self.updateInfoCheckTarget(agent, timeStep)
                return 0

        elif action == 0:
            self.rewards[agent.ID - 1] = reward.NOMOVE.value
            self.updateInfoCheckTarget(agent, timeStep)
            return 0

        else:
            print("INVALID ACTION: {}".format(action))
            sys.exit()

        newLoc = agent.getLocation()
        self.worldState[origLoc[0]][origLoc[1]] = 0
        self.worldState[newLoc[0]][newLoc[1]] = agent.ID
        if action <= 4:
            self.rewards[agent.ID - 1] = reward.MOVE.value
        elif action > 4:
            self.rewards[agent.ID - 1] = reward.MOVEDIAGONAL.value
        self.updateInfoCheckTarget(agent, timeStep)
        return action

    def updateInfoCheckTarget(self, agent, timeStep):
        """
        update the self.infoMap and check whether the agent has found a target
        """
        transform_row = self.observationSize // 2 - agent.row
        transform_col = self.observationSize // 2 - agent.col

        min_row = max((agent.row - self.observationSize // 2), 0)
        max_row = min((agent.row + self.observationSize // 2 + 1), self.shape[0])
        min_col = max((agent.col - self.observationSize // 2), 0)
        max_col = min((agent.col + self.observationSize // 2 + 1), self.shape[1])

        # build a observe filter matrix###
        filter = np.full((max_row - min_row, max_col - min_col),0)

        for i in range(min_row, max_row):
            for j in range(min_col, max_col):
                if self.region_map[i][j] == agent.region_number:
                    filter[i - min_row][j - min_col] = 1
                else:
                    filter[i - min_row][j - min_col] = 0

        infoMap_temp = np.array(self.infoMap)
        uncertainty_temp = np.array(self.uncertaintyMap)

        # update the whole world infoMap and uncertaintyMap with the filter_SenseArea
        # update agents' individual maps
        self.uncertaintyMap[min_row:max_row, min_col:max_col] *= 1 - (agent.SenseArea[(min_row + transform_row):(max_row + transform_row),
                                                                                      (min_col + transform_col):(max_col + transform_col)]*filter)
        updateSense = (self.infoMap[min_row:max_row, min_col:max_col] + uncertainty_temp[min_row:max_row,
                                                                        min_col:max_col]) * \
                      (1 - (agent.SenseArea[(min_row + transform_row):(max_row + transform_row),
                            (min_col + transform_col):(max_col + transform_col)]*filter))
        self.infoMap[min_row:max_row, min_col:max_col] = np.minimum(infoMap_temp[min_row:max_row, min_col:max_col],
                                                                    updateSense)
        self.infoMap[self.infoMap < 0] = 0
        self.uncertaintyMap[self.uncertaintyMap < 0] = 0
        agent.updateUncertainty()
        agent.updateInfo()
        if self.model == 'A':
            agent.updatefromglobal(self.infoMap, self.uncertaintyMap)
            agent.individual_targetMap = self.targetMap
        # print(infoMap_temp, "\n")
        amountInfo = np.sum(infoMap_temp - self.infoMap) / self.totalInfo
        amountUncertainty = np.sum(uncertainty_temp - self.uncertaintyMap) / self.totalUncertainty
        # random the reward of cleaning the uncertainty
        total_amount = amountUncertainty * np.random.rand() + amountInfo
        agent.PercentageInfo = amountInfo
        agent.PercentageUncertainty = amountUncertainty
        # print("uncertainty", amountUncertainty, "Dust", amountDust)
        self.rewards[agent.ID - 1] += (1 / self.removeSpeed0) * total_amount * reward.REMOVEDUST.value
        # print(self.rewards)

        # check the target
        ObserveTargetMap = np.full((self.observationSize, self.observationSize), 0)
        ObserveTargetMap[(min_row + transform_row):(max_row + transform_row),
        (min_col + transform_col):(max_col + transform_col)] = self.targetMap[min_row:max_row,
                                                               min_col:max_col]

        targetPos = np.where(ObserveTargetMap < 0)

        for i in range(len(targetPos[0])):
            targetRow = targetPos[0][i]
            targetCol = targetPos[1][i]

            if agent.region_number != self.region_map[targetRow - transform_row][targetCol - transform_col]:
                continue
            # check the targets
            if np.random.rand() < agent.SenseArea[targetRow][targetCol]:
                agent.individual_targetMap[targetRow - transform_row][targetCol - transform_col] = -3
                # if the target haven't been found
                if ObserveTargetMap[targetRow][targetCol] == -2:
                    self.rewards[agent.ID - 1] += reward.FINDTARGET.value
                    self.targetMap[targetRow - transform_row][targetCol - transform_col] = -3
                for targetID in range(1, len(self.targets) + 1):
                    if self.targets[targetID - 1].row == targetRow - transform_row \
                            and self.targets[targetID - 1].col == targetCol - transform_col:
                        self.targets[targetID - 1].updateFound(timeStep)
                        self.targets[targetID - 1].status[agent.ID - 1] = True
                        self.agents[agent.ID - 1].target_status[targetID - 1] = self.targets[targetID - 1].status

    # Execute one time step within the environment
    def step(self, agentID, action, timeStep):
        """
        the agents execute the actions
        No movement: 0
        North (-1,0): 1
        East (0,1): 2
        South (1,0): 3
        West (0,-1): 4
        """

        assert (agentID > 0)

        with self.mutex:
            actualAction = self.executeAction(self.agents[agentID - 1], action, timeStep)

            self.finished |= self.check_finish()

        return self.agents[agentID - 1].PercentageInfo, self.agents[agentID - 1].PercentageUncertainty

    def observe(self, agentID):
        assert (agentID > 0)
        if len(self.targets) != np.sum(self.targetMap == -2) + np.sum(self.targetMap == -3):
            import pickle
            bugoutput = {"worldState": self.worldState, "targetMap": self.targetMap, "targets": self.targets,
                         "agents": self.agents, "infos": self.infoMap}
            pickle.dump(bugoutput, open('annoyingbug1.mat', 'wb'))
            print("\t\tBug happened AGAIN")
            # assert(len(self.targets) == np.sum(self.targetMap == -2) + np.sum(self.targetMap == -3))

        # scalarObs = [ np.sum(self.targetMap == -2), self.agents[agentID - 1].row / (self.shape[0]-1), self.agents[agentID - 1].col / (self.shape[1]-1) ]
        scalarObs = [self.agents[agentID - 1].row / (self.shape[0] - 1),
                     self.agents[agentID - 1].col / (self.shape[1] - 1)]

        vectorObs = self.extractObservation(self.agents[agentID - 1])
        if self.model == "B":
            self.updateMap(agentID)
        # pyplot.imshow(self.agents[agentID - 1].individual_uncertainty)
        # pyplot.show()
        return [vectorObs, scalarObs], self.rewards[agentID - 1]

    def targetScan(self):
        """
        scan all the target and find the time for the first target found and the half targets found
        """
        TimeFound = []
        for num in range(len(self.targets)):
            TimeFound.append(self.targets[num].time_found)
        TimeFound = np.sort(np.array(TimeFound))
        # first = TimeFound[0]
        # half = TimeFound[int(len(self.targets) // 2 + 1)]
        if not np.isnan(TimeFound[0]):
            first = TimeFound[0]
        else:
            first = 256
        if not np.isnan(TimeFound[int(len(self.targets) // 2)]):
            half = TimeFound[int(len(self.targets) // 2)]
        else:
            half = 256

        return first, half

    def check_finish(self):
        status = []
        for t in self.targets:
            status.append(t.status)
        status = np.sum(status, axis=1)
        d = np.all(status)
        return d

    def reset(self, state0=None, targetMode=0):
        """
        reset the world using the self.setWorld
        """
        self.finished = False

        # Initialize data structures
        self.setWorld(state0, targetMode=targetMode)

        if self.viewer is not None:
            self.viewer = None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def render(self, mode='human', agentID=7, close=False, screenWidth=850, screenHeight=850):
        # Render the environment to the screen
        if self.viewer is None:
            self.viewer = rendering.Viewer(screenWidth, screenHeight)

        rows = self.shape[0]
        cols = self.shape[1]
        size = 400 / max(rows, cols)
        interval = 450
        # agentID = agentID

        infoMap = np.rot90(self.infoMap, k=3)
        uncertaintyMap = np.rot90(self.uncertaintyMap, k=3)
        worldState = np.rot90(self.worldState, k=3)
        targetMap = np.rot90(self.targetMap, k=3)
        agent_targetMap = np.rot90(self.agents[agentID - 1].individual_targetMap, k=3)
        agent_info = np.rot90(self.agents[agentID - 1].individual_info, k=3)
        agent_uncertain = np.rot90(self.agents[agentID - 1].individual_uncertainty, k=3)
        maxInfo = self.maxInfo
        minInfo = self.minInfo
        colorlen_info = maxInfo - minInfo
        maxUncertianty = self.maxUncertainty
        minUncertainty = self.minUncertainty
        colorlen_uncertainty = maxUncertianty - minUncertainty
        # render the boundary line ###
        boundary_line = np.rot90(self.boundary_line, k=3)

        for i in range(cols):
            for j in range(rows):
                # rending the infoMap
                if agent_info[i, j] != 0:
                    color = (self.highestProbColor[0],
                             self.lowestProbColor[1] - (agent_info[i, j] - minInfo) / colorlen_info,
                             self.lowestProbColor[2] - (agent_info[i, j] - minInfo) / colorlen_info)
                    self.viewer.add_onetime(create_rectangle(i * size, j * size, size, size, color, False))
                elif agent_info[i, j] == 0:
                    self.viewer.add_onetime(create_rectangle(i * size, j * size, size, size, self.lowestProbColor, False))

                # rending the target
                if agent_targetMap[i, j] == -2:
                    self.viewer.add_onetime(create_rectangle(i * size, j * size, size, size, self.targetNotFound, False))
                elif agent_targetMap[i, j] == -3:
                    self.viewer.add_onetime(create_rectangle(i * size, j * size, size, size, self.targetFound, False))

                # rending the agent and the obstacle
                if worldState[i, j] == -1:
                    self.viewer.add_onetime(create_rectangle(i * size, j * size, size, size, self.obstacleColor, False))
                elif worldState[i, j] == agentID:
                    self.viewer.add_onetime(create_circle(i * size, j * size, size, size, self.agentColor))

                # rending the boundary lines
                if boundary_line[i, j] == 0:
                    self.viewer.add_onetime(create_rectangle(i * size, j * size, size, size, self.boundarycolor, False))

        for i in range(cols):
            for j in range(rows):
                # rending the uncertaintyMap
                if agent_uncertain[i, j] != 0:
                    color = (self.lowestProbColor[0] - (agent_uncertain[i, j] - minUncertainty) / colorlen_uncertainty,
                             self.lowestProbColor[1] - (agent_uncertain[i, j] - minUncertainty) / colorlen_uncertainty,
                             self.highestUncertaintyColor[2])
                    self.viewer.add_onetime(create_rectangle(i * size + interval, j * size, size, size, color, False))
                elif agent_uncertain[i, j] == 0:
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size, size, size, self.lowestProbColor, False))

                # rending the target
                if agent_targetMap[i, j] == -2:
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size, size, size, self.targetNotFound, False))
                elif agent_targetMap[i, j] == -3:
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size, size, size, self.targetFound, False))

                # rending the agent and the obstacle
                if worldState[i, j] == -1:
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size, size, size, self.obstacleColor, False))
                elif worldState[i, j] == agentID:
                    self.viewer.add_onetime(create_circle(i * size + interval, j * size, size, size, self.agentColor))

                # rending the boundary lines
                if boundary_line[i, j] == 0:
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size, size, size, self.boundarycolor, False))

        for i in range(cols):
            for j in range(rows):
                # rending the infoMap
                if infoMap[i, j] != 0:
                    color = (self.highestProbColor[0],
                             self.lowestProbColor[1] - (infoMap[i, j] - minInfo) / colorlen_info,
                             self.lowestProbColor[2] - (infoMap[i, j] - minInfo) / colorlen_info)
                    self.viewer.add_onetime(create_rectangle(i * size, j * size + interval, size, size, color, False))
                elif infoMap[i, j] == 0:
                    self.viewer.add_onetime(
                        create_rectangle(i * size, j * size + interval, size, size, self.lowestProbColor, False))

                # rending the target
                if targetMap[i, j] == -2:
                    self.viewer.add_onetime(
                        create_rectangle(i * size, j * size + interval, size, size, self.targetNotFound, False))
                elif targetMap[i, j] == -3:
                    self.viewer.add_onetime(
                        create_rectangle(i * size, j * size + interval, size, size, self.targetFound, False))

                # rending the boundary lines
                if boundary_line[i, j] == 0:
                    self.viewer.add_onetime(
                        create_rectangle(i * size, j * size + interval, size, size, self.boundarycolor, False))

                # rending the agent and the obstacle
                if worldState[i, j] == -1:
                    self.viewer.add_onetime(
                        create_rectangle(i * size, j * size + interval, size, size, self.obstacleColor, False))
                elif worldState[i, j] > 0:
                    if self.agents[int(worldState[i, j]) - 1].communicate_activated:
                        self.viewer.add_onetime(
                            create_circle(i * size, j * size + interval, size, size, self.agentCommColor))
                    else:
                        self.viewer.add_onetime(create_circle(i * size, j * size + interval, size, size, self.agentColor))


        for i in range(cols):
            for j in range(rows):
                # rending the uncertaintyMap-whole
                if uncertaintyMap[i, j] != 0:
                    color = (self.lowestProbColor[0] - (uncertaintyMap[i, j] - minUncertainty) / colorlen_uncertainty,
                             self.lowestProbColor[1] - (uncertaintyMap[i, j] - minUncertainty) / colorlen_uncertainty,
                             self.highestUncertaintyColor[2])
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size + interval, size, size, color, False))
                elif uncertaintyMap[i, j] == 0:
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size + interval, size, size, self.lowestProbColor, False))

                # rending the target
                if targetMap[i, j] == -2:
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size + interval, size, size, self.targetNotFound, False))
                elif targetMap[i, j] == -3:
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size + interval, size, size, self.targetFound, False))

                # rending the boundary lines
                if boundary_line[i, j] == 0:
                    self.viewer.add_onetime(create_rectangle(i * size + interval, j * size + interval, size, size, self.boundarycolor,False))

                # rending the agent and the obstacle
                if worldState[i, j] == -1:
                    self.viewer.add_onetime(
                        create_rectangle(i * size + interval, j * size + interval, size, size, self.obstacleColor, False))
                elif worldState[i, j] > 0:
                    if self.agents[int(worldState[i, j]) - 1].communicate_activated:
                        self.viewer.add_onetime(
                            create_circle(i * size + interval, j * size + interval, size, size, self.agentCommColor))
                        self.agents[int(worldState[i, j]) - 1].communicate_activated = False
                    else:
                        self.viewer.add_onetime(
                            create_circle(i * size + interval, j * size + interval, size, size, self.agentColor))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


if __name__ == "__main__":
    # test for plot
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    # world0 = np.asarray([
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  6.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  8.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  6.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  3.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  4.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  1.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  2., 0.,  5.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  7.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=np.float64)
    # targetMap = np.asarray([
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  6.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0., -2.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0., -2.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0., -2.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -2.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.],
    #     [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=np.float64)
    # uncertainty = np.ones((20, 20)) / 400.
    # state0 = np.asarray([np.ones((20, 20)) / 400., uncertainty, world0, targetMap])

    state0 = None
    # state0 = np.load("/home/marmot/Weiheng/DRL_A3C_MASearch/environments/1.npy")
    obs_density = 0.
    env = MAS_gym(state0=state0, obs_density=obs_density, model="B", shapes=(64, 64))

    for i in range(100):
        for a in range(1, 9):
            env.step(action=np.random.randint(1, 8), agentID=a, timeStep=50)
            env.communicate(agentID=a)
        a, b = env.observe(agentID=7)
        #print(b, "\n")
        # print(env.agents[6].individual_info)
        env.render()
        time.sleep(0.01)
    # print(env.targetMap)
    # print(env.worldState)
    #print(env.targets[0].time_found)
    #print(env.targetScan())

    # fig1 = plt.figure()
    # ax = fig1.gca(projection='3d')
    # M, N = np.meshgrid(np.linspace(0, env.shape[0] - 1, env.shape[0]),
    #                    np.linspace(0, env.shape[1] - 1, env.shape[1]))
    # surf = ax.plot_surface(M, N, env.infoMap, cmap=cm.gist_rainbow)
    # fig1.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
    #
    # fig2 = plt.figure()
    # ax = fig2.gca(projection='3d')
    # M, N = np.meshgrid(np.linspace(0, env.shape[0] - 1, env.shape[0]),
    #                    np.linspace(0, env.shape[1] - 1, env.shape[1]))
    # surf = ax.plot_surface(M, N, env.uncertaintyMap, cmap=cm.gist_rainbow)
    # fig2.colorbar(surf, shrink=0.5, aspect=5)
    # plt.show()
