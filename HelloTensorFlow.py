# encode : utf-8
import tensorflow as tf
print("Result is : ")
hi = tf.constant("Hello TensorFlow World", shape=[1,1])
print("Befor Session hi is : ", end="")
print(hi)
print(hi.shape)
    
with tf.Session() as sess:
        
     hi = sess.run(hi)
     print("After Session hi is : ", end="")
     print(hi)
     print(hi.shape)
      

"""
    Result is :
	Befor Session hi is : Tensor("Const:0", shape=(1, 1), dtype=string)
	(1, 1)

	After Session hi is : [[b'Hello TensorFlow World']]
	(1, 1)
    

---------------------

本文来自 小宋shine 的CSDN 博客 ，全文地址请点击：https://blog.csdn.net/xiaosongshine/article/details/82847725?utm_source=copy 
"""
