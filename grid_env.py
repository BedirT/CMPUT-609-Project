'''
. . . D1| . . .P1
. . . . | . . . .
. . . . | . . . .
G . . . | . . . .
________    _____
S . . . | . . . .
. . . . | . . . .
. . . . | . . . .
D2. . .   . . .P2

It's a gridworld environment with stochasticity implemented.
S is the starting point, and agent can move in cardinal directions,
the goal is to reach the point G. There is no direct way to get into
the room of G, so the agent needs to use one of the portals to teleport
inside the room. Points P are the portal locations and D's are the 
destinations. Stochasticity is here, that everytime when a portal is used
the destinations of the portals are renewed. One portal is matched with
the destination in the same room as G and the other is the one in the room
of S. The probabilities are almost random (or will be random depending on
the results), starting with %60 D1 for the P1 and %40 D1 for the P2.
(Trying to see the effect of the distance)
'''
from numpy import random
import numpy as np

class grid_environment:

    def __init__(self):
        self.reset()
        self.action_space = np.array([0, 1, 2, 3])
  
    def start(self):
        self.reset()
        return self.agent_pos

    def one_hot(self, p):
        a = np.zeros(len(self.the_world[0]) * len(self.the_world))
        a[p[0] * len(self.the_world[0]) + p[1]] = 1
        return a.reshape((a.shape[0], 1))

    def step(self, action):
        reward = 0
        done = False
        # self.reward_t = self.roomLoc()
        # pos[1]: x
        # pos[0]: y
        if action == 0: # w: x-1
            if self.agent_pos[1] - 1 < 0 or self.the_world[self.agent_pos[0]][self.agent_pos[1]-1] == '|'\
                or self.the_world[self.agent_pos[0]][self.agent_pos[1]-1] == '-':
                # if we hit a wall or out of boundry
                reward = self.reward_b + self.reward_t
            else:
                # valid move
                self.update_agents_pos(self.agent_pos[1]-1,self.agent_pos[0])
                self.agent_pos[1] -= 1
                reward = self.reward_t
        elif action == 1: # e: x+1
            if self.agent_pos[1] + 1 >= len(self.the_world[1]) or self.the_world[self.agent_pos[0]][self.agent_pos[1]+1] == '|'\
                or self.the_world[self.agent_pos[0]][self.agent_pos[1]+1] == '-':
                # if we hit a wall or out of boundry
                reward = self.reward_b + self.reward_t
            else:
                # valid move
                self.update_agents_pos(self.agent_pos[1]+1,self.agent_pos[0])
                self.agent_pos[1] += 1
                reward = self.reward_t
        elif action == 2: # s: y+1
            if self.agent_pos[0] + 1 >= len(self.the_world[0]) or self.the_world[self.agent_pos[0]+1][self.agent_pos[1]] == '-'\
                or self.the_world[self.agent_pos[0]+1][self.agent_pos[1]] == '|':
                # if we hit a wall or out of boundry
                reward = self.reward_b + self.reward_t
            else:
                # valid move
                self.update_agents_pos(self.agent_pos[1],self.agent_pos[0]+1)
                self.agent_pos[0] += 1
                reward = self.reward_t
        elif action == 3: # n: y-1
            if self.agent_pos[0] - 1 < 0 or self.the_world[self.agent_pos[0]-1][self.agent_pos[1]] == '-'\
                or self.the_world[self.agent_pos[0]-1][self.agent_pos[1]] == '|':
                # if we hit a wall or out of boundry
                reward = self.reward_b + self.reward_t
            else:
                # valid move
                self.update_agents_pos(self.agent_pos[1],self.agent_pos[0]-1)
                self.agent_pos[0] -= 1
                reward = self.reward_t

        if self.agent_pos == self.goal_pos: # reached the goal
            done = True
            self.the_world[self.goal_pos[0]][self.goal_pos[1]] = 'G'
            reward += self.reward_g
        elif self.agent_pos == self.p1_location: # portal 1   
            self.update_agents_pos(self.p1_des[0], self.p1_des[1])
            self.agent_pos = self.p1_des
            self.the_world[self.p1_location[0]][self.p1_location[1]] = 'P1'
            self.choose_portals()
        elif self.agent_pos == self.p2_location: # portal 2
            self.update_agents_pos(self.p2_des[0], self.p2_des[1])
            self.agent_pos = self.p2_des
            self.the_world[self.p2_location[0]][self.p2_location[1]] = 'P2'
            self.choose_portals()
        
        return reward, self.agent_pos, done

    ##### HELPER FUNCTIONS ######
    def roomLoc(self):
        room1 = []
        for i in range(5):
            for j in range(5):
                room1.append([i,j])
        if self.agent_pos in room1:
            return -3
        return -1

    def choose_portals(self):
        if random.choice([i for i in range(10)]) < 6: 
            self.p1_des, self.p2_des = self.d1_location, self.d2_location
        else:
            self.p1_des, self.p2_des = self.d2_location, self.d1_location
        
    def update_agents_pos(self, x, y):
        self.the_world[self.agent_pos[0]][self.agent_pos[1]] = '.'
        self.the_world[y][x] = 'A'

    def print_board(self):
        for i in range(len(self.the_world)):
            for j in range(len(self.the_world[i])):
                print(self.the_world[i][j], end='  ')
            print()   

    def reset(self):
        # self.the_world = [
        #     ['.', '.', '.', 'D1', '|', '.', '.', '.', 'P1'],
        #     ['.', '.', '.', '.' , '|', '.', '.', '.', '.'],
        #     ['.', '.', '.', '.' , '|', '.', '.', '.', '.'],
        #     ['G', '.', '.', '.' , '.', '.', '.', '.', '.'],
        #     ['-', '-', '-', '-' , '|', '.', '-', '-', '-'],
        #     ['A', '.', '.', '.' , '|', '.', '.', '.', '.'],
        #     ['.', '.', '.', '.' , '|', '.', '.', '.', '.'],
        #     ['.', '.', '.', '.' , '|', '.', '.', '.', '.'],
        #     ['D2','.', '.', '.' , '.', '.', '.', '.', 'P2']
        # ]

        self.the_world = [
            ['.', 'D1', '|', '.', 'P1'],
            ['G', '.' , '|', '.', '.'],
            ['-', '-' , '|', '.', '-'],
            ['A', '.' , '|', '.', '.'],
            ['D2','.' , '.', '.', 'P2'],
        ]

        # self.the_world = [
        #     ['.','.', '.', '.' , '.', '.', '.', '.', 'G'],
        #     ['.','.', '.', '.' , '.', '.', '.', '.', '.'],
        #     ['.','.', '.', '.' , '.', '.', '.', '.', '.'],
        #     ['.','.', '.', '.' , '.', '.', '.', '.', '.'],
        #     ['.','.', '.', '.' , '.', '.', '.', '.', '.'],
        #     ['.','.', '.', '.' , '.', '.', '.', '.', '.'],
        #     ['.','.', '.', '.' , '.', '.', '.', '.', '.'],
        #     ['.','.', '.', '.' , '.', '.', '.', '.', '.'],
        #     ['A','.', '.', '.' , '.', '.', '.', '.', '.']
        # ]

        # self.agent_start_pos = [8, 0]
        self.agent_start_pos = [3, 0]
        self.agent_pos = self.agent_start_pos

        self.reward_b = 0 # reward when hitting the walls or boundries 
        self.reward_t = -1 # reward per time step
        self.reward_g = 10 # reward if the agent gets to the goal

        self.p1_location = [0, 4]
        self.p2_location = [4, 4]

        self.d1_location = [0, 1]
        self.d2_location = [4, 0]
        
        self.goal_pos = [1, 0]
        # self.goal_pos = [0, 8]
        
        self.grid_size = len(self.the_world)
 
        self.choose_portals()