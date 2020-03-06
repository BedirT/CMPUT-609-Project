import gym
import numpy as np
import matplotlib.pyplot as plt
import copy 
from grid_env import grid_environment
from tile_coding import tile_coding
from REINFORCE import agent_REINFORCE
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
params = {
    'alg' : 'SARSA',
    'num_of_episodes' : 1000,
    'max_steps' : 1000,
    'num_of_runs' : 20,
    'num_of_parameters_to_test' : 5,
    'alpha' : 0.00001,
    'gamma' : 0.98,
    'seed_num' : 1,
    # Creating the tilings
    'grid_size' : 5,
    'tile_size' : 2,
    'num_of_tiles' : 2
}

# to Render the board on terminal or not
play_it_per = 100
play_it = False

tilings = tile_coding(params['grid_size'], params['num_of_tiles'], params['tile_size'], 4)
# print(tilings.num_of_tilings)
# exit()
 
env = grid_environment()
np.random.seed(params['seed_num'])

# Keep stats for final print of graph
episode_rewards = np.zeros((params['num_of_parameters_to_test'], params['num_of_runs'], params['num_of_episodes']))
step_took = np.zeros(params['num_of_episodes'])

if params['alg'] == 'REINFORCE':
    agent = agent_REINFORCE(tilings.num_of_tilings, env.action_space.shape[0], params['alpha'], params['gamma'])
    # params['alpha'] = 2 ** (-13)
else:
    agent = SG_SARSA(tilings.num_of_tilings, env.action_space.shape[0], params['alpha'], params['gamma'])
    # params['alpha'] = 0.00001

for p in tqdm.tqdm(range(params['num_of_parameters_to_test'])):
    params['alpha'] = 2 ** (-p-10)
    agent.alpha = params['alpha']

    for r in range(params['num_of_runs']):
        params['seed_num'] = r+1
        np.random.seed(params['seed_num'])
        
        agent.reset_weights()

        for ep in range(params['num_of_episodes']):

            state = tilings.active_tiles(env.start()) # a x d

            grads = []
            rewards = []
            states = []
            actions = []

            score = 0
            step_ = 0
            # while True:
            for _ in range(params['max_steps']):
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
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))

the_time = str(time.time()) + params['alg']
Path(dir_path + "/saves/" + the_time).mkdir(parents=True, exist_ok=True)

np.save(dir_path + "/saves/"+ the_time +"/rewards", episode_rewards)
np.save(dir_path + "/saves/"+ the_time +"/params", params)
#############

plt.plot(episode_rewards[0][0]/params['num_of_runs'])
plt.show()