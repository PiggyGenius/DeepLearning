import tensorflow as tf
import numpy as np

# Executing variables and operators by their names is very useful in many scenarios. For example, we may develop a model in a separate module; and thus the variables are not available in a different Python scope according to Python scoping rules. However, if we have a graph, we can execute the nodes of the graph using their names in the graph.
# This can be done easily by changing the sess.run method from the previous code example, using the variable name of the cost in the graph rather than the Python variable cost by changing sess.run([cost, train_op], ...) to sess.run(['cost:0', 'train_op'], ...)
# Tensorflow appends ':0' to the name of the first tensor, '_1:0' to the second and so on...
n_epochs = 500
training_costs = []
with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(n_epochs):
        c, _ = sess.run(['cost: 0', 'train_op'], feed_dict={'tf_x:0': x_train, 'tf_y:0': y_train})
        training_costs.append(c)
        if e % 50 == 0:
            print('Epoch {:4d}: {:.4f}'.format(e, c))


# In the previous section, we built a graph and trained it. How about doing the actual prediction on the held out test set? The problem is that we did not save the model parameters; so, once the execution of the preceding statements are finished and we exit the tf.Session environment, all the variables and their allocated memories are freed.
# One solution is to train a model, and as soon as the training is finished, we can feed it our test set. However, this is not a good approach since deep neural network models are typically trained over multiple hours, days, or even weeks.
# The best approach is to save the trained model for future use
