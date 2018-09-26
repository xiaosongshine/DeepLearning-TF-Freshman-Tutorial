# encoding: utf-8
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

in_units = 784
h1_units = 64
out_units = 10

"""
Tensors常量值函数

tf.zeros(shape, dtype=tf.float32, name=None)
tf.zeros_like(tensor, dtype=None, name=None)
tf.ones(shape, dtype=tf.float32, name=None)
tf.ones_like(tensor, dtype=None, name=None)
tf.fill(dims, value, name=None)
tf.constant(value, dtype=None, shape=None, name='Const')

---------------------

本文来自 长风o 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/Hk_john/article/details/78189676?utm_source=copy 
"""

w1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.constant(.1, dtype=tf.float32, shape=[1, h1_units])) #b1 = tf.Variable(tf.fill([h1_units, 1], 0.1))
w2 = tf.Variable(tf.truncated_normal([h1_units, out_units], stddev=0.1))
b2 = tf.Variable(tf.constant(.1, dtype=tf.float32, shape=[1, out_units]))


"""
tf.placeholder(dtype, shape=None, name=None)
"""
x = tf.placeholder(tf.float32, [None, in_units])
keep_prob = tf.placeholder(tf.float32)

y1 = tf.matmul(x, w1) + b1
hidden1 = tf.nn.relu(y1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)
y2 = tf.matmul(hidden1_drop, w2) + b2
y = tf.nn.softmax(y2)

y_ = tf.placeholder(tf.float32, [None, out_units])
"""
tf.reduce_sum(
    input_tensor,
    axis=None,
    keepdims=None,
    name=None,
    reduction_indices=None,
    keep_dims=None
)
Args:

input_tensor: The tensor to reduce. Should have numeric type. #输入
	axis: The dimensions to reduce. If None (the default), reduces all dimensions. Must be in the range [-rank(input_tensor), rank(input_tensor)).
	keepdims: If true, retains reduced dimensions with length 1.
	name: A name for the operation (optional).
	reduction_indices: The old (deprecated) name for axis.
	keep_dims: Deprecated alias for keepdims.

---------------------
"""
y = tf.clip_by_value(y, 0.001, 1.999)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))
train_step = tf.train.AdagradOptimizer(0.2).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    start_time = time.time()
    for steps in range(5000):

        batch_xs, batch_ys = mnist.train.next_batch(100)
        cross_entropy_val, train_step_val = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.75})

        if(steps % 100 == 0):
            end_time = time.time()
            print("steps is %d loss is %f every step use tims is %f" % (steps, cross_entropy_val, end_time - start_time), end="")
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_val = sess.run(accuracy, feed_dict={x: mnist.test.images[:500], y_:  mnist.test.labels[:500], keep_prob: 1.})
            print(" Accuracy is %f"%accuracy_val)
            start_time = time.time()
            pass

    accuracy_val = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.})
    print("Final Accuracy is %f" % accuracy_val)

