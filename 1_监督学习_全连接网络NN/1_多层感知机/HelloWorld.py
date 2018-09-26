# encoding:utf-8

import tensorflow as tf

hi = tf.constant("Hello world", shape=[1, 1])

print("Befor Session:hi is ", end="")
print(hi)
print(hi.shape)

with tf.Session() as sess:
    hi = sess.run(hi)
    print("After Session:hi is ", end="")
    print(hi)
    print(hi.shape)




