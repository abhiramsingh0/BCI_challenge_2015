import tensorflow as tf
import numpy as np
sess = tf.InteractiveSession()
a=tf.Variable(5)
b=a+5
sess.run(tf.global_variables_initializer())
print sess.run(b)
a=b
print sess.run(a)
print sess.run(b)
