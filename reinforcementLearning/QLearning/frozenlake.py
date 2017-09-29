import gym
import numpy as np

# We load the fronzeLake environment
env = gym.make('FrozenLake-v0')
# We initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# We set the learning parameters
lr = 0.8
y = 0.95
num_episodes = 2000

# We create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # We reset the environment and get the first new observation
    s = env.reset()
    rAll = 0
    d = False
    j = 0
    # We use the Q-table learning algorithm
    print('\nStart\n')
    while j < 99:
        # env.render()
        j += 1
        # We choose the action by greedily picking from Q table with noise
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        # We get new state and reward from environment
        s1, r, done, _ = env.step(a)
        # Update Q-table with new knowledge
        Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
        rAll += r
        s = s1
        if done == True:
            break
    rList.append(rAll)
print('Score over time: ' + str(sum(rList) / num_episodes))
print('Final Q-table values')
print(Q)
