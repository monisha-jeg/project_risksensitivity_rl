import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.display import display, HTML
pd.set_option("display.precision", 2)

action_map = {0: "up", 1:"down", 2:"right", 3:"left"}
symbol_map = [u"\u2b06", u"\u2b07", u"\u27a1", u"\u2b05"]


class gridworld:

	def __init__(self, dims, goals, error_states, p, gen_reward, goal_reward, error_reward):
		#M - rows
		self.M = dims[0]
		#N - columns
		self.N = dims[1]
		#current state (initial) - start state
		self.current_state = None
		#goal state
		self.goals = goals
		#error_states - Error state indices
		self.error_states = error_states
		#p - probability of moving in the direction of action
		self.p = p
		#np - probability of moving in a direction other than that of action		
		self.np = (1.0 - p)/3
		#reward for reaching a general state
		self.gen_reward = gen_reward
		#reward for reaching an error state
		self.error_reward = error_reward
		#reward for reaching a goal state
		self.goal_reward = goal_reward

		# #visual representation of gridworld
		self.visual_gw = np.array([["-" for i in range(self.N)] for j in range(self.M)])
		for g in self.goals:
			self.visual_gw[g] = "G"
		for es in self.error_states:
			self.visual_gw[es] = "E"

		self.num_good_states = self.M *self.N - (len(self.goals) + len(self.error_states))
		self.enum_good_states = []
		for i in range(self.M):
			for j in range(self.N):
				if (i, j) not in self.goals + self.error_states:
					self.enum_good_states.append((i, j))





	#set current state to random state
	def reset(self):
		#reset to random state
		rannum = np.random.choice(self.num_good_states)
		self.current_state = self.enum_good_states[rannum]
		return self.current_state



	#input - action
	#output - new state, reward
	def step(self, action_num):

		action = action_map[action_num]
		alternate_moves = { "up" : ["down", "right", "left"],
							"down" : ["up", "right", "left"],
							"left" : ["down", "right", "up"],
							"right" : ["down", "up", "left"],
							}

		#sample direction of movement according to action and p
		rand_num = np.random.rand(1)[0]
		if rand_num < self.p:
			move = action
		elif rand_num < self.p + self.np:
			move = alternate_moves[action][0]
		elif rand_num < self.p + 2*self.np:
			move = alternate_moves[action][1]
		else:
			move = alternate_moves[action][2]

		#find new state using state and move
		new_state = (-1, -1)
		if move == "up":
			if self.current_state[0] == 0:
				new_state = self.current_state	
			else:
				new_state = (self.current_state[0]-1, self.current_state[1])
		elif move == "down":
			if self.current_state[0] == self.M-1:
				new_state = self.current_state
			else:
				new_state = (self.current_state[0]+1, self.current_state[1])
		elif move == "left":
			if self.current_state[1] == 0:
				new_state = self.current_state
			else:
				new_state = (self.current_state[0], self.current_state[1]-1)
		elif move == "right":
			if self.current_state[1] == self.N-1:
				new_state = self.current_state
			else:
				new_state = (self.current_state[0], self.current_state[1]+1)

		self.current_state = new_state
		

		#return new state, reward and a boolean representing whether a terminal state is done or not
		if new_state in self.goals:
			return new_state, self.goal_reward, True
		elif new_state in self.error_states:
			return new_state, self.error_reward, True
		else:
			return new_state, self.gen_reward, False


	def set_state(self, state):
		self.current_state = state


	#print gridworld
	def print_gw(self):				
		display(pd.DataFrame(self.visual_gw))


	#print value
	def print_v(self, v):
		vis_v = v.tolist()

		for g in self.goals:
			vis_v[g[0]][g[1]] = "G"
		for e in self.error_states:
			vis_v[e[0]][e[1]] = "E"

		vis_v = pd.DataFrame(vis_v)
		display(vis_v)
		



	#print policy
	def print_policy(self, pi):
		vis_pi = pi.tolist()

		for i in range(pi.shape[0]):
			for j in range(pi.shape[1]):
				vis_pi[i][j] = symbol_map[pi[i, j]]
		for g in self.goals:
			vis_pi[g[0]][g[1]] = "G"
		for e in self.error_states:
			vis_pi[e[0]][e[1]] = "E"

		vis_pi = pd.DataFrame(vis_pi)
		display(vis_pi)






# gw = gridworld((5, 5), [(0, 0), (4, 4)], [], 0.9, 0, 1, 0)
# gw.reset()
# gw.print_gw()
# gw.step(1)
# gw.print_gw()
# gw.step(1)
# gw.print_gw()








		





