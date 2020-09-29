import numpy as np
np.seterr(divide='ignore', invalid='ignore')
from MAS_gym.envs.MAS_env import MAS_gym
from multiprocessing import Pool

# OPPOSITE_ACTIONS = {1: 3, 2: 4, 3: 1, 4: 2, 0: 0, 5: 7, 7: 5, 6: 8, 8: 6}

def asStride(arr,sub_shape,stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0,s1=arr.strides[:2]
    m1,n1=arr.shape[:2]
    m2,n2=sub_shape
    view_shape=(1+(m1-m2)//stride[0],1+(n1-n2)//stride[1],m2,n2)+arr.shape[2:]
    strides=(stride[0]*s0,stride[1]*s1,s0,s1)+arr.strides[2:]
    subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
    return subs

def pooling(mat,ksize,stride=None,method='max',pad=False):
    '''Overlapping pooling on 2D or 3D data.
    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <method>: str, 'max for max-pooling,
                   'mean' for mean-pooling.
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).
    Return <result>: pooled matrix.
    '''

    m, n = mat.shape[:2]
    ky,kx=ksize
    if stride is None:
        stride=(ky,kx)
    sy,sx=stride

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,sy)
        nx=_ceil(n,sx)
        size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]

    view=asStride(mat_pad,ksize,stride)

    if method=='max':
        result=np.nanmax(view,axis=(2,3))
    else:
        result=np.nanmean(view,axis=(2,3))

    return result


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
        a = observation[2]  # info
        b = observation[3]  # unc
        c = observation[0]  # nearby agents
        # actionid = 0

        # Make info & unc cells with low value as 0
        a[a < 0.0002] = 0.0
        b[b < 0.0002] = 0.0

        # Center square from 11x11
        a_11x11 = a[4:7, 4:7]
        b_11x11 = b[4:7, 4:7]
        m_11x11 = np.array((a_11x11, b_11x11))

        # Center square from 9x9
        a_9x9 = pooling(a, (3, 3), stride=(1, 1), method='mean', pad=False)
        a_9x9 = a_9x9[3:6, 3:6]
        b_9x9 = pooling(b, (3, 3), stride=(1, 1), method='mean', pad=False)
        b_9x9 = b_9x9[3:6, 3:6]
        m_9x9 = np.array((a_9x9, b_9x9))

        # Center square from 6x6
        a_6x6 = pooling(a, (6, 6), stride=(1, 1), method='mean', pad=False)
        a_6x6 = a_6x6[1:4, 1:4]
        b_6x6 = pooling(b, (6, 6), stride=(1, 1), method='mean', pad=False)
        b_6x6 = b_6x6[1:4, 1:4]
        m_6x6 = np.array((a_6x6, b_6x6))

        # Center square from 3x3
        a_3x3 = pooling(a, (5, 5), stride=(3, 3), method='max', pad=False)
        b_3x3 = pooling(b, (5, 5), stride=(3, 3), method='max', pad=False)
        m_3x3 = np.array((a_3x3, b_3x3))

        # Merging multiScales with weights
        m = m_3x3 * 0.1 + m_6x6 * 0.4 + m_9x9 * 0.2 + m_11x11 * 0.3
        a = 0.6 * m[0] + 0.4 * m[1]

        # adx, ady = np.gradient(
        #    np.array([[a[3, 3], a[3, 5], a[3, 7]], [a[5, 3], a[5, 5], a[5, 7]], [a[7, 3], a[7, 5], a[7, 7]]]))
        adx, ady = np.gradient(a)
        infovec = np.array([adx[1, 1], ady[1, 1]]) / np.linalg.norm(np.array([adx[1, 1], ady[1, 1]]))
        # infovec = np.array([adx[5, 5], ady[5, 5]]) / np.linalg.norm(np.array([adx[5, 5], ady[5, 5]]))
        # bdx, bdy = np.gradient(
        #    np.array([[b[3, 3], b[3, 5], b[3, 7]], [b[5, 3], b[5, 5], b[5, 7]], [b[7, 3], b[7, 5], b[7, 7]]]))
        # bdx, bdy = np.gradient(m[1])
        # uncvec = np.array([bdx[1, 1], bdy[1, 1]]) / np.linalg.norm(np.array([bdx[1, 1], bdy[1, 1]]))
        # uncvec = np.array([bdx[5, 5], bdy[5, 5]]) / np.linalg.norm(np.array([bdx[5, 5], bdy[5, 5]]))
        agentvec = []
        for i in range(0, self.grid_size):
            for j in range(0, self.grid_size):
                if c[i, j] > 0 and i != 5 and j != 5:
                    agentvec.append(np.array([5 - i, 5 - j]) / np.linalg.norm(np.array([5 - i, 5 - j])))
        if len(agentvec) == 0:
            direction = infovec / np.linalg.norm(infovec)
        else:
            agentvec = np.mean(agentvec, 0) / np.linalg.norm(np.mean(agentvec, 0))
            direction = (0.6 * infovec + 0.4 * agentvec) / np.linalg.norm(
                0.6 * infovec + 0.4 * agentvec)

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
    gym_grad = MAS_gym(numAgents=8, model="B", shapes=(64, 64),REGION = None, AGENT = None, BOUNDARY = None , INFO = None, TARGET= None)
    grad.set_env(gym_grad)
    grad_info, image = grad.find_path(256)
    return grad_info

def main():
    totalMap = 8
    totalTest = 1
    worldSize = [64]
    num_agents = 4
    taskList = []
    while num_agents < 8:
        num_agents = int(num_agents * 2)
        for size in worldSize:
            if size == 20 and num_agents > 32:
                continue
            if size == 40 and num_agents > 64:
                continue
            if size == 80 and num_agents > 128:
                continue
            for testID in range(totalTest):
                for mapID in range(totalMap):
                    taskList.append((num_agents, size, mapID, testID))

    num_threads = 1
    print("task list generated")
    np.random.shuffle(taskList)
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