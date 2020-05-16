from gridworld import *
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras import backend as K


#REINFORCE

#gw = gridworld((4, 4), (0, 0), (3, 3), [], 0.9, 0, 100, -100)
#gw = gridworld((4, 4), (0, 0), (3, 3), [(0, 3), (1, 3), (2, 3)], 0.9, 0, 100, 0)
#gw = gridworld((5, 5), (0, 0), (2, 2), [(2, 1), (1, 2)], 0.9, -0.1, 0.9, -1.1)
gw = gridworld((5, 5), (0, 4), (4, 4), [(o, 4) for o in range(1, 4)], 0.9, 0, 1, 0)
y = 0.95

#Initialize table with all zeros
Q = np.zeros((gw.M,gw.N,4))

# Set learning parameters
learning_rate = 0.1
global_step = 0
EPISODES = 2000
scores = []
#create lists to contain total rewards and steps per episode

samples_states, samples_actions, samples_rewards = [], [], []


def one_hot_state(state):
    zeros = np.zeros((1, gw.M*gw.N))
    zeros[0][state[0]*gw.M + state[1]] = 1
    return zeros

def one_hot_action(action):
    zeros = np.zeros((4))
    zeros[action] = 1
    return zeros


#model
#layers
model = Sequential()
model.add(Dense(10, input_dim=gw.M*gw.N, activation='relu'))
# model.add(Dense(10, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()
action = K.placeholder(shape=[None, 4])
discounted_rewards = K.placeholder(shape=[None, ])
# Calculate cross entropy error function
action_prob = K.sum(action * model.output, axis=1)
cross_entropy = K.log(action_prob) * discounted_rewards
loss = -K.sum(cross_entropy)
# create training function
optimizer = Adam(lr=learning_rate)
updates = optimizer.get_updates(model.trainable_weights, [], loss)
train = K.function([model.input, action, discounted_rewards], [], updates=updates)


#choose action in simulation
def get_action(state):
    policy = model.predict(state)[0]
    #print(policy)
    return np.random.choice(4, 1, p=policy)[0]


# calculate discounted rewards
def discount_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(0, len(rewards))):
        running_add = running_add * y + rewards[t]
        discounted_rewards[t] = running_add
    discounted_rewards = np.float32(discounted_rewards)
    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    #print(discounted_rewards)
    return discounted_rewards



# save states, actions and rewards for an episode
def append_sample(state, action, reward):
    global samples_rewards, samples_states, samples_actions
    samples_states.append(state[0])
    samples_rewards.append(reward)
    samples_actions.append(one_hot_action(action))    



def train_model():
    global samples_rewards, samples_actions, samples_states
    discounted_rewards = np.float32(discount_rewards(samples_rewards))
    #print(discounted_rewards)    

    #print(samples_states, samples_actions, discounted_rewards)
    train([samples_states, samples_actions, discounted_rewards])
    samples_states, samples_actions, samples_rewards = [], [], []


for e in range(EPISODES):
    print("Episode", e)
    done = False
    score = 0
    # fresh env
    state = one_hot_state(gw.reset())

    while not done:
        global_step += 1
        # get action for the current state and go one step in environment
        action = get_action(state)
        next_state, reward, done = gw.step(action)
        next_state = one_hot_state(next_state)

        append_sample(state, action, reward)
        score += reward
        state = next_state

        if done:
            # update policy neural network for each episode
            train_model()
            scores.append(score)


pi = []
for i in range(gw.M):
    row = []
    for j in range(gw.N):
        action = np.argmax(model.predict(one_hot_state((i, j))))
        row.append(action)    
    pi.append(row)
gw.print_policy(np.array(pi))

# v = np.mean(Q, axis = 2)
# gw.print_v(v)

# pi = np.argmax(Q, axis = 2)

