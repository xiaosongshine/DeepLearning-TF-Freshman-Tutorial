# encoding : utf-8
import time
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("fashion/", one_hot=True)
keep_prob = tf.placeholder(tf.float32)
train = True


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x, size):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1],
                          strides=[1, size, size, 1], padding='SAME')


# 第0层输入层
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# def AlexNet(x_, train):
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 第一层卷积层 28*28*1 -->
W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool(h_conv1, 2)
if (train == True):
    h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

    # 第二次卷积层
W_conv2 = weight_variable([3, 3, 64, 128])
b_conv2 = bias_variable([128])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool(h_conv2, 2)
if (train == True):
    h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

    # 第三层卷积池化层
W_conv3 = weight_variable([3, 3, 128, 256])
b_conv3 = bias_variable([256])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool(h_conv3, 2)
if (train == True):
    h_pool3 = tf.nn.dropout(h_pool3, keep_prob)

    # print(h_pool3.shape)
h_pool3_flat = tf.reshape(h_pool3, [-1, 4 * 4 * 256])

W_fc1 = weight_variable([4 * 4 * 256, 1024])
b_fc1 = bias_variable([1024])

h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1 = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

y_conv = tf.clip_by_value(y_conv, 0.00001, 1.9999)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    start_time = time.time()
    for steps in range(4000):

        batch_xs, batch_ys = mnist.train.next_batch(50)
        cross_entropy_val, train_step_val = sess.run([cross_entropy, train_step],
                                                     feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.8})

        if (steps % 100 == 0):
            end_time = time.time()
            print("steps is %d loss is %f ,100 step use tims is %f sec" % (
            steps, cross_entropy_val, end_time - start_time), end="")
            # y_conv1 = AlexNet(x, False)
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_val = sess.run(accuracy,
                                    feed_dict={x: mnist.test.images[:100], y_: mnist.test.labels[:100], keep_prob: 1.})
            print(" Accuracy is %f" % accuracy_val)
            start_time = time.time()
            pass

    accuracy_val = sess.run(accuracy, feed_dict={x: mnist.test.images[1000: 2000], y_: mnist.test.labels[1000: 2000], keep_prob: 1.})
    print("Final Accuracy is %f" % accuracy_val)



