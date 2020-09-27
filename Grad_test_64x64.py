import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from MAS_gym.envs.MAS_env import MAS_gym
from multiprocessing import Pool

# OPPOSITE_ACTIONS = {1: 3, 2: 4, 3: 1, 4: 2, 0: 0, 5: 7, 7: 5, 6: 8, 8: 6}
class MASearch_grad(object):
    '''
    This class provides functionality for running multiple instances of the
    trained network in a single environment
    '''

    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.actionlist = []
        self.OPPOSITE_ACTIONS = {1: 3, 2: 4, 3: 1, 4: 2, 0: 0, 5: 7, 7: 5, 6: 8, 8: 6}
    def set_env(self, gym):
        self.num_agents = gym.numAgents
        self.size = gym.shape
        self.env = gym
        self.actionlist = []

    def gradVec(self, observation, agent):
        a = observation[2]
        b = observation[3]
        c = observation[0]
        actionid = 0

        adx, ady = np.gradient(
            np.array([[a[3, 3], a[3, 5], a[3, 7]], [a[5, 3], a[5, 5], a[5, 7]], [a[7, 3], a[7, 5], a[7, 7]]]))
        # adx, ady = np.gradient(a)
        infovec = np.array([adx[1, 1], ady[1, 1]]) / np.linalg.norm(np.array([adx[1, 1], ady[1, 1]]))
        # infovec = np.array([adx[5, 5], ady[5, 5]]) / np.linalg.norm(np.array([adx[5, 5], ady[5, 5]]))
        bdx, bdy = np.gradient(
            np.array([[b[3, 3], b[3, 5], b[3, 7]], [b[5, 3], b[5, 5], b[5, 7]], [b[7, 3], b[7, 5], b[7, 7]]]))
        # bdx, bdy = np.gradient(b)
        uncvec = np.array([bdx[1, 1], bdy[1, 1]]) / np.linalg.norm(np.array([bdx[1, 1], bdy[1, 1]]))
        # uncvec = np.array([bdx[5, 5], bdy[5, 5]]) / np.linalg.norm(np.array([bdx[5, 5], bdy[5, 5]]))
        agentvec = []
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if c[i, j] > 0 and i != 5 and j != 5:
                    agentvec.append(np.array([5 - i, 5 - j]) / np.linalg.norm(np.array([5 - i, 5 - j])))
        if len(agentvec) == 0:
            direction = 1 * infovec + 0 * uncvec / np.linalg.norm(0.8 * infovec + 0 * uncvec)
        else:
            agentvec = np.mean(agentvec, 0) / np.linalg.norm(np.mean(agentvec, 0))
            direction = 0.7 * infovec + 0 * uncvec + 0.2 * agentvec / np.linalg.norm(
                0.6 * infovec + 0 * uncvec + 0.2 * agentvec)

        action_vec = [[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1], [-1, -1], [-1, 1], [1, 1], [1, -1]]
        actionid = np.argmax([np.dot(direction, a) for a in action_vec])
        actionid = self.check_valid_action(actionid, agent, direction)
        return actionid

    def check_valid_action(self, actionid, agent, direction):
        if len(self.actionlist) > 1:
            if actionid == self.OPPOSITE_ACTIONS[self.actionlist[self.step - 1][agent - 1]]:
                action_vec = [[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1], [-1, -1], [-1, 1], [1, 1], [1, -1]]
                actionid = np.array([np.dot(direction, a) for a in action_vec])
                actionid = actionid.argsort()[7]
        return actionid

    def step_all_parallel(self):
        action_probs = [None for i in range(self.num_agents)]
        '''advances the state of the environment by a single step across all agents'''
        # parallel inference
        actions = []
        reward = 0
        for agent in range(1, self.num_agents + 1):
            self.env.communicate(agent)
        for agent in range(1, self.num_agents + 1):
            o, r = self.env.observe(agent)
            reward += r
            actions.append(self.gradVec(o[0], agent))
        self.actionlist.append(actions)
        for agent in range(1, self.num_agents + 1):
            self.env.step(agent, actions[agent - 1], self.step)
        return reward

    def find_path(self, max_step=256):
        '''run a full environment to completion, or until max_step steps'''
        TimeFound = []
        saveimage = True
        self.step = 0
        frames = []
        reward = 0
        # if np.random.rand() > 0.98:
        #     saveimage = True
        while (not self.env.finished) and self.step < max_step:
            # timestep = []
            # for agent in range(1, self.env.numAgents + 1):
            #     timestep.append(self.env.agents[agent - 1].getLocation())
            if saveimage:
                frames.append(self.env.render(mode='rgb_array'))
            r = self.step_all_parallel()
            reward += r
            self.step += 1
        # if self.step == max_step:
        #     raise OutOfTimeError
        if saveimage:
            frames.append(self.env.render(mode='rgb_array'))
        first, half = self.env.targetScan()
        # for num in range(len(self.env.targets)):
        #     TimeFound.append(self.env.targets[num].time_found)
        # TimeFound = np.sort(np.array(TimeFound))
        episodeInfo = [self.env.communicateTimes / 2, self.env.newCommunicateTimes / 2,
                       (self.env.totalInfo - np.sum(
                           self.env.infoMap)) / self.env.totalInfo / self.num_agents / self.step,
                       (self.env.totalUncertainty - np.sum(
                           self.env.uncertaintyMap)) / self.env.totalUncertainty / self.num_agents / self.step,
                       first,half,reward,self.step]
        return episodeInfo, frames

def testing(episode):
    grad = MASearch_grad(11)
    gym_grad = MAS_gym(numAgents=8, model="B", shapes=(64, 64),REGION = None, AGENT = None, BOUNDARY = None , INFO = None)
    grad.set_env(gym_grad)
    grad_info, image = grad.find_path(256)
    return grad_info

def main():
    episode = 1
    num_threads = 16
    print("task list generated")
    taskList = [i for i in range(episode)]
    p = Pool(num_threads)
    episode_info = p.map(testing, taskList)
    print("finished all tests!")
    episode_info = np.array(episode_info)
    episode_info = np.mean(episode_info, axis=0, dtype=float)
    print("Comm: ", int(episode_info[0]))
    print("New comm: ", int(episode_info[1]))
    print("Per_info: ", episode_info[2])
    print("Per_unc: ", episode_info[3])
    print("First: ", int(episode_info[4]))
    print("Half : ", int(episode_info[5]))
    print("Reward: ", episode_info[6])
    print("Length: ", int(episode_info[7]))


if __name__ == "__main__":
    main()