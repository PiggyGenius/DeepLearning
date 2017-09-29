import numpy as np
import tensorflow as tf

# We create four bandits that return a normal rand value of mean 0
bandits = [0.2, 0, -0.2, -5]
num_bandits = len(bandits)
def pull_bandits(bandit):
    result = np.random.randn(1)
    if result > bandit:
        return 1
    else:
        return -1

# We have a set of values for each bandits, estimates of the return value if chosen
tf.reset_default_graph()
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights, 0)

# We define the training procedure
reward_holder = tf.placeholder(shape=[1], dtype=tf.float32)
action_holder = tf.placeholder(shape=[1], dtype=tf.int32)
responsible_weight = tf.slice(weights, action_holder, [1])
loss = -(tf.log(responsible_weight) * reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)

# We train the agent
total_episodes = 1000
total_reward = np.zeros(num_bandits)
e = 0.1
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(total_episodes):
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        reward = pull_bandits(bandits[action])
        # We update the network
        _, resp, ww = sess.run([update, responsible_weight, weights], feed_dict = {
            reward_holder: [reward],
            action_holder: [action]
        });
        total_reward[action] += reward
agent = np.argmax(ww)
print('\nThe agent thinks bandit {} is the most promising.'.format(str(agent + 1))) 
if agent == np.argmax(-np.array(bandits)):
    print('The agent was right.')
else:
    print('The agent was wrong.')
