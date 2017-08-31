#from sklearn import metrics
import tensorflow as tf
#from  read_proces_data import *
import numpy as np
sess = tf.InteractiveSession()
x = np.ones((60,3260,57),dtype=np.float32)
size = 128
num_layers = 4
def lstm_cell(): 
    return tf.contrib.rnn.BasicLSTMCell(size,state_is_tuple=True)
cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
output, state = tf.nn.dynamic_rnn(cell,x,dtype=tf.float32)
sess.run(tf.global_variables_initializer())
#print output.eval()
print cell.output_size
