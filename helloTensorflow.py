# encode : utf-8

import tensorflow as tf

print("Result is : ")
hi = tf.constant("Hello TensorFlow World", shape=[1,1])
print("Befor Session hi is : ", end="")
print(hi)

with tf.Session() as sess:
    
    hi = sess.run(hi)
    print("After Session hi is : ", end="")
    print(hi)
  
"""
Result is:
Befor Session hi is : Tensor("Const:0", shape=(1, 1), dtype=string)

2018-09-25 23:48:18.641193: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2

After Session hi is : [[b'Hello TensorFlow World']]

"""
