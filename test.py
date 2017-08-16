#import tensorflow as tf
import numpy as np
from read_proces_data import *
iter_over_files =  1
#for itera in range(0, iter_over_files):
#    for file_ in file_dir:
#        print file_
#        # get features from current indexed file
#        features = get_train_data(file_)
#        # get target values corresponding to this file
#        target_values = get_target_values(file_)
#        # training features is np array with max size and padding
#        training_features, seq_len = prepare_data(features)
#        print training_features.shape
#        print len(seq_len)

for itera in range(0, iter_over_files):
    for file_ in test_file_dir:
        print file_
        # get features from current indexed file
        features = get_test_data(file_)
        # get target values corresponding to this file
        #target_values = get_target_values(file_)
        # training features is np array with max size and padding
        training_features, seq_len = prepare_data(features)
        print training_features.shape
        print len(seq_len)















#sess = tf.InteractiveSession()
#x = np.random.rand(60,2378,57)
#w = np.ones((128,1))
#x = tf.cast(x,tf.float32)
#w = tf.cast(w,tf.float32)
#hidden_units = 128
#lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units =\
#        hidden_units,state_is_tuple=True)
#a,b= lstm_cell.zero_state(60,tf.float32)
#print a
#print b
#c = tf.zeros([60,hidden_units])
#h = tf.zeros([60,hidden_units])
##a = [60, s] for s in lstm_cell.state_size
#i_state = tf.nn.rnn_cell.LSTMStateTuple(a,b)
#output_, state = tf.nn.dynamic_rnn(\
#        lstm_cell,\
#        x,\
#        initial_state = i_state,\
#        dtype=tf.float32)
##y =  output_[:,0,:]
##yy = tf.matmul(y,w)
#sess.run(tf.global_variables_initializer())
##index = np.random.randint(0,2378,(60),dtype=np.int32)
##a = output_[:,index[0],:]
#c,h= state
#print c
#print h
#sess.run(state)
##print w[0,:]
##print w[1,:]
