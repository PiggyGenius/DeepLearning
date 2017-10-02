import tensorflow as tf
import numpy as np

##################################################################################
################ We can transform tensors as multidimensional array ##############
##################################################################################

g = tf.Graph()
with g.as_default():
    arr = np.array([[1., 2., 3., 3.5], [4., 5., 6., 6.5], [7., 8., 9., 9.5]])
    T1 = tf.constant(arr, name='T1')
    s = T1.get_shape()
    T2 = tf.Variable(tf.random_normal(shape=s))
    T3 = tf.Variable(tf.random_normal(shape=(s.as_list()[0], )))
    print(T1)
    print(T2)
    print(T3)
    print('Shape of T1 is', s)
# We can reshape tensors
# -1 means that the size will be infered based on the size of the array and the specified dimensions
with g.as_default():
    T4 = tf.reshape(T1, shape=[1, 1, -1], name='T4')
    T5 = tf.reshape(T1, shape=[1, 3, -1], name='T5')
    print(T4)
    print(T5)
with tf.Session(graph=g) as sess:
    print(sess.run(T4))
    print(sess.run(T5))
# We can transpose and change the order of dimensions with perm
with g.as_default():
    T6 = tf.transpose(T5, perm=[2, 1, 0], name='T6')
    T7 = tf.transpose(T5, perm=[0, 2, 1], name='T7')
    print(T6)
    print(T7)
# We can split a tensor into a list of subtensors
with g.as_default():
    t5_splt = tf.split(T5, num_or_size_splits=2, axis=2, name='T8')
    print(t5_splt)
# We can combine a list of tensors with the same shape and dtype
g = tf.Graph()
with g.as_default():
    t1 = tf.ones(shape=(5, 1), dtype=tf.float32, name='t1')
    t2 = tf.zeros(shape=(5, 1), dtype=tf.float32, name='t2')
    print(t1)
    print(t2)
with g.as_default():
    t3 = tf.concat([t1, t2], axis=0, name='t3')
    t4 = tf.concat([t1, t2], axis=1, name='t4')
    print(t3)
    print(t4)
with tf.Session(graph=g) as sess:
    print(t3.eval())
    print(t4.eval())


##################################################################################
######################## Control flow mechanics in graphs ########################
##################################################################################

# We implement this equation in Tensorflow: res = x + y if x < y and x - y otherwise
# This is incorrect, the computation graph has only one branch associated with the addition operator
# The computation graph is static, once the graph is built it remains unchanged during the execution
# Even when we change the values, the new tensors will go through the same path in the graph
x, y = 1.0, 2.0
g = tf.Graph()
with g.as_default():
    tf_x = tf.placeholder(dtype=tf.float32, shape=None, name='tf_x')
    tf_y = tf.placeholder(dtype=tf.float32, shape=None, name='tf_y')
    if x < y:
        res = tf.add(tf_x, tf_y, name='result_add')
    else:
        res = tf.substract(tf_x, tf_y, name='result_sub')
    print('Object:', res)
with tf.Session(graph=g) as sess:
    print('x < y: {} -> Result:'.format(x < y), res.eval(feed_dict={'tf_x:0': x, 'tf_y:0': y}))
    x, y = 2.0, 1.0
    print('x < y: {} -> Result:'.format(x < y), res.eval(feed_dict={'tf_x:0': x, 'tf_y:0': y}))
# Now let's implement it correctly using control flow mechanics in TensorFlow
x, y = 1.0, 2.0
g = tf.Graph()
with g.as_default():
    tf_x = tf.placeholder(dtype=tf.float32, shape=None, name='tf_x')
    tf_y = tf.placeholder(dtype=tf.float32, shape=None, name='tf_y')
    res = tf.cond(tf_x < tf_y, 
            lambda: tf.add(tf_x, tf_y, name='result_add'), 
            lambda: tf.subtract(tf_x, tf_y, name='result_sub'))
    print('Object:', res)
with tf.Session(graph=g) as sess:
    print('x < y: {} -> Result:'.format(x < y), res.eval(feed_dict={'tf_x:0': x, 'tf_y:0': y}))
    x, y = 2.0, 1.0
    print('x < y: {} -> Result:'.format(x < y), res.eval(feed_dict={'tf_x:0': x, 'tf_y:0': y}))
# To implement this statement: if (x < y) result = 1 else result = 0
f1 = lambda: tf.constant(1)
f2 = lambda: tf.constant(0)
result = tf.case([(tf.less(x, y), f1)], default=f2)
# We can also do a while loop
i = tf.constant(0)
threshold = 100
c = lambda i: tf.less(i, 100)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(cond=c, body=b, loop_vars=[i])
