import gym
import numpy as np
import matplotlib.pyplot as plt
import copy 
from grid_env import grid_environment
from tile_coding import tile_coding
from REINFORCE import agent_REINFORCE
from REINFORCEv2 import agent_REINFORCE as REINFORCEv2
from Semi_Gradient_SARSA import SG_SARSA
import os, tqdm, time

'''
Hyperparameters to play with:
num_of_episodes:
max_steps:
alpha:
gamma:
seed_num:

tile_size:
num_of_tiles:
'''
alg = 'REINFORCE'
# alg = 'SARSA'

num_of_episodes = 5000
max_steps = 1000
num_of_runs = 1
num_of_parameters_to_test = 1

alpha = 0.5/8
gamma = 0.98
seed_num = 1

# to Render the board on terminal or not
play_it_per = 100
play_it = False

# Creating the tilings
grid_size = 5
tile_size = 2
num_of_tiles = 2
tilings = tile_coding(grid_size, num_of_tiles, tile_size, 4)
# print(tilings.num_of_tilings)
# exit()
 
env = grid_environment()
np.random.seed(seed_num)

# Keep stats for final print of graph
episode_rewards = np.zeros((num_of_parameters_to_test, num_of_runs, num_of_episodes))
step_took = np.zeros(num_of_episodes)

if alg == 'REINFORCE':
    agent = agent_REINFORCE(tilings.num_of_tilings, env.action_space.shape[0], alpha, gamma)
else:
    agent = SG_SARSA(tilings.num_of_tilings, env.action_space.shape[0], alpha, gamma)

for p in tqdm.tqdm(range(num_of_parameters_to_test)):
    agent.alpha = 2 ** (-p-10)

    for r in tqdm.tqdm(range(num_of_runs)):
        np.random.seed(r+1)
        agent.reset_weights()

        for ep in range(num_of_episodes):

            state = tilings.active_tiles(env.start()) # a x d

            grads = []
            rewards = []
            states = []
            actions = []

            score = 0
            step_ = 0
            # while True:
            for _ in range(max_steps):
                if alg == 'REINFORCE':
                    action = agent.step(state)
                else:
                    action = agent.step(state)
                reward, next_state, done = env.step(action)
                next_state = tilings.active_tiles(next_state)
                
                rewards.append(reward)
                states.append(state)
                actions.append(action)
                
                score += reward
                step_ += 1
                state = next_state

                if play_it and ep % play_it_per == 0:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    env.print_board()

                if done:
                    break

            agent.update(states, actions, rewards)
                
            episode_rewards[p][r][ep] = score
            print("EP: ", ep, " Score: ", score, "         ",end="\r", flush=False)
        # plt.plot(episode_rewards[p]/num_of_runs, label="a = 2^" +str(-p-10))


###### SAVING
dir_path = os.path.dirname(os.path.realpath(__file__))
if alg == 'REINFORCE':
    np.save(dir_path +"/saves/theta/"+ alg + '_imp_' + str(time.time()), agent.theta)
else:
    np.save(dir_path +"/saves/w/"+ alg + '_' + str(time.time()), agent.w)
np.save(dir_path +"/saves/rewards/"+ alg + '_imp_'+ str(num_of_runs) +'_numruns_' + str(time.time()), episode_rewards)
#############

# for i in range(num_of_runs):
#     # print(alpha_rewards[i])
#     plt.plot(alpha_rewards[i])
# plt.show()

plt.plot(episode_rewards[0][0]/num_of_runs)
plt.show()