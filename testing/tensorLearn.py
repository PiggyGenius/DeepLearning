import tensorflow as tf
import numpy as np

#####################################################################
######################### Ranks and tensors #########################
#####################################################################

# tf.rank() --> tensor: get the rank of a tensor by evaluating the result
# X.get_shape() --> TensorShape: if X is a tensor
# We can print it but no index or slice the object, has to use X.as_list()
g = tf.Graph()
with g.as_default():
    t1 = tf.constant(np.pi)
    t2 = tf.constant([1, 2, 3, 4])
    t3 = tf.constant([[1, 2], [3, 4]])
    r1 = tf.rank(t1)
    r2 = tf.rank(t2)
    r3 = tf.rank(t3)
    s1 = t1.get_shape()
    s2 = t2.get_shape()
    s3 = t3.get_shape()
    print('Shapes: ', s1, s2, s3)
with tf.Session(graph=g) as sess:
    print('Ranks:', r1.eval(), r2.eval(), r3.eval())


######################################################################
############################ Placeholders ############################
######################################################################

# We define a graph for evaluating z = 2 * (a - b) + c
g = tf.Graph()
with g.as_default():
    # If shape was 2x3x4, we would have shape=[2, 3, 4]
    tf_a = tf.placeholder(tf.int32, shape=[], name='tf_a')
    tf_b = tf.placeholder(tf.int32, shape=[], name='tf_b')
    tf_c = tf.placeholder(tf.int32, shape=[], name='tf_c')
    r1 = tf_a - tf_b
    r2 = 2 * r1
    z = r2 + tf_c
# We feed the values of placeholders with data arrays, every placeholder has to be fed
with tf.Session(graph=g) as sess:
    feed = {tf_a: 1, tf_b: 2, tf_c: 3}
    print('z:', sess.run(z, feed_dict=feed))

# We can train a neural network with a set mini-batch size and make predictions on more data input
# We can create a placeholder of rank 2 where the first dimension is unknown or may vary
g = tf.Graph()
with g.as_default():
    tf_x = tf.placeholder(tf.float32, shape=[None, 2], name='tf_x')
    x_mean = tf.reduce_mean(tf_x, axis=0, name='mean')
# We can evaluate x_mean with different input, x1 and x2 which are of shape (5, 2) and (10, 2)
np.random.seed(123)
np.set_printoptions(precision=2)
with tf.Session(graph=g) as sess:
    x1 = np.random.uniform(low=0, high=1, size=(5, 2))
    print('Feeding data with shape', x1.shape)
    print('Result:', sess.run(x_mean, feed_dict={tf_x: x1}))
    x2 = np.random.uniform(low=0, high=1, size=(10, 2))
    print('Feeding data with shape', x2.shape)
    print('Result:', sess.run(x_mean, feed_dict={tf_x: x2}))
print(tf_x)


#####################################################################
############################# Variables #############################
#####################################################################

# Variables allow us to store and update the parameters of our model
# Variables store the parameters of a model that can be updated during training
# Tensors defined as variables are not allocated in memory and contain no values until initialized
g1 = tf.Graph()
with g1.as_default():
    w = tf.Variable(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]), name='w')
    print(w)
with tf.Session(graph=g1) as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(w))
# We can store the operator and execute later: init = tf.gl_va_in();sess.run(init) || init.run
g2 = tf.Graph()
with g2.as_default():
    w1 = tf.Variable(1, name='w1')
    init_op = tf.global_variables_initializer()
    w2 = tf.Variable(2, name='w2')
# We can evaluate w1 but not w2
with tf.Session(graph=g2) as sess:
    sess.run(init_op)
    print('w1:', sess.run(w1))
# with tf.Session(graph=g2) as sess:
    # sess.run(init_op)
    # print('w2:', sess.run(w2))
# With variable scopes, we can organize the variables into seperate subparts
# In a variable scope, the name of operations and tensors that are created are prefixed in it
# If we have 2 subnetworks, each has several layers, each layer can be defined in a scope
g = tf.Graph()
with g.as_default():
    with tf.variable_scope('net_A'):
        with tf.variable_scope('layer-1'):
            w1 = tf.Variable(tf.random_normal(shape=(10, 4)), name='weights')
        with tf.variable_scope('layer-2'):
            w2 = tf.Variable(tf.random_normal(shape=(20, 10)), name='weights')
    with tf.variable_scope('net_B'):
        with tf.variable_scope('layer-1'):
            w3 = tf.Variable(tf.random_normal(shape=(10, 4)), name='weights')
    print(w1)
    print(w2)
    print(w3)

# We assume that we have data (X_a, y_a) coming from source A and data (X_b, y_b) from source B
# We use the data from only one source as input tensor to build the network
# Then we can feed the data from the other source to the same classifier
# We assume that data from A is fed through placegolder and B is the output of a generator network
def build_classifier(data, labels, n_classes=2):
    data_shape = data.get_shape().as_list()
    weights = tf.get_variable(name='weights', shape=(data_shape[1], n_classes), dtype=tf.float32)
    bias = tf.get_variable(name='bias', initializer=tf.zeros(shape=n_classes))
    logits = tf.add(tf.matmul(data, weights), bias, name='logits')
    return logits, tf.nn.softmax(logits)
def build_generator(data, n_hidden):
    data_shape = data.get_shape().as_list()
    w1 = tf.Variable(tf.random_normal(shape=(data_shape[1], n_hidden)), name='w1')
    b1 = tf.Variable(tf.zeros(shape=n_hidden), name='b1')
    hidden = tf.add(tf.matmul(data, w1), b1, name='hidden_pre-activation')
    hidden = tf.nn.relu(hidden, 'hidden_activation')
    w2 = tf.Variable(tf.random_normal(shape=(n_hidden, data_shape[1])), name='w2')
    b2 = tf.Variable(tf.zeros(shape=data_shape[1]), name='b2')
    output = tf.add(tf.matmul(hidden, w2), b2, name='output')
    return output, tf.nn.sigmoid(output)

batch_size = 64
g = tf.Graph()
with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), dtype=tf.float32, name='tf_X')
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)
    with tf.variable_scope('classifier') as scope:
        cls_out1 = build_classifier(data=tf_X, labels=tf.ones(shape=batch_size))
        scope.reuse_variables()
        cls_out2 = build_classifier(data=gen_out1[1], labels=tf.zeros(shape=batch_size))
        init_op = tf.global_variables_initializer()
# We called build_classifier 2 times, the first creates the variables and the second reuse them
# We can also do it by specifying reuse=True
g = tf.Graph()
with g.as_default():
    tf_X = tf.placeholder(shape=(batch_size, 100), dtype=tf.float32, name='tf_X')
    with tf.variable_scope('generator'):
        gen_out1 = build_generator(data=tf_X, n_hidden=50)
    with tf.variable_scope('classifier'):
        cls_out1 = build_classifier(data=tf_X, labels=tf.ones(shape=batch_size))
    with tf.variable_scope('classifier', reuse=True):
        cls_out2 = build_classifier(data=gen_out1[1], labels=tf.zeros(shape=batch_size))
