import numpy as np
from MAS_gym.envs.MAS_env import *
from Grad_test_64x64 import *

# input variables
REGION = np.load('region_map.npy')
AGENT = np.load('agent_map.npy')
INFO = np.load('info.npy')
LINE = np.load('line.npy')
TARGET = np.load('target.npy')

def testing(episode):
    grad = MASearch_grad(11)
    gym_grad = MAS_gym(numAgents=8, model="B", shapes=(64, 64),REGION = REGION, AGENT = AGENT, BOUNDARY = LINE , INFO = INFO, TARGET= TARGET)
    grad.set_env(gym_grad)
    grad_info, image = grad.find_path(256)
    return grad_info

def search():
    episode = 1
    num_threads = 16
    taskList = [i for i in range(episode)]
    p = Pool(num_threads)
    episode_info = p.map(testing, taskList)
    episode_info = np.array(episode_info)
    episode_info = np.mean(episode_info, axis=0, dtype=float)
    # print("Length: ", int(episode_info[7]))

    return episode_info[7]

if __name__ == "__main__":
    hjg = search()
    print(hjg)
