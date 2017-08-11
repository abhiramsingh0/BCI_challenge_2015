#import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
from  read_proces_data import *
sess = tf.InteractiveSession()

# create lstm cell in tensorflow
hidden_units = 256
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units =\
        hidden_units,state_is_tuple=True)
# define initial state of lstm
#c_h = tf.placeholder(tf.string)
#h_h = tf.placeholder(tf.string)

#c_h, c = tf.get_session_tensor(c_h, tf.float32)
#h_h, h = tf.get_session_tensor(h_h, tf.float32)

#c_,h_ = lstm_cell.zero_state(60,tf.float32)
#i_state = tf.contrib.rnn.LSTMStateTuple(c,h)
batch_size = 100
x = tf.placeholder(tf.float32, shape = [batch_size,2378,57])
s_len = tf.placeholder(tf.int32,shape=[batch_size])
# run lstm over different sequence length
output_, state = tf.nn.dynamic_rnn(\
        lstm_cell,\
        x,\
#        initial_state = i_state,\
        sequence_length=s_len,\
        dtype=tf.float32)
# take all batch, last output of each batch and full output vector
def take_subarray(array , index):
    return array[range(0,batch_size),index-1]
output = tf.py_func(take_subarray,[output_,s_len],tf.float32)
#print output
# define final output value
target_value = tf.placeholder(tf.float32, shape=[batch_size])
# define weights and bias from output of lstm cell to network final output
rv = tf.truncated_normal([lstm_cell.output_size, 1],\
        stddev=0.1,dtype=tf.float32)
W = tf.Variable(rv,dtype=tf.float32)
rb = tf.constant(0.1,shape=[batch_size,1],dtype=tf.float32)
b = tf.Variable(rb,dtype=tf.float32)
# initialize variables
# calculate final output
logits = tf.matmul(output, W) + b
observed = tf.sigmoid(logits)

# calculate cost as calculated in logistic classification.
cost = tf.reduce_mean(-1.0 * target_value * tf.log(observed) - \
        (1 - target_value) * tf.log(1 - observed))
#cost = tf.reduce_mean(-1 * target_value * tf.log(observed))
#cost = tf.reduce_mean(tf.square(target_value - observed))
# optimize the cost
train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
def threshold_fn(array):
    return array >= 0.50
obser = tf.py_func(threshold_fn,[observed], tf.bool)
observe = tf.cast(obser,tf.float32)
correct_prediction = tf.equal(observe, target_value)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
"""
o = np.asarray(output_)
o = tf.reshape(o[-1,-1,:],[128,1])
linear_output = tf.m)
"""
#c_,h_ = lstm_cell.zero_state(60,tf.float32)
#c_handle = tf.get_session_handle(c_)
#c_handle = sess.run(c_handle)
#h_handle = tf.get_session_handle(h_)
#h_handle = sess.run(h_handle)

# number of times to iterate over whole training dataset
iter_over_files =  1
for itera in range(0, iter_over_files):
    for file_ in file_dir:
        # get features from current indexed file
        features = get_train_data(file_)
        # get target values corresponding to this file
        target_values = get_target_values(file_)
        # training features is np array with max size and padding
        training_features, seq_len = prepare_data(features)

        for i in range(0,1000):
            tr_fe, sl , tv =\
            sample_data(training_features,target_values,seq_len,batch_size)
#            if 0==i:
            sess.run(train_step, feed_dict = {target_value:tv,\
                    x:tr_fe, s_len:sl})
#                    c_h:c_handle.handle,h_h:h_handle.handle})
#
#                c__, h__ = state
#                c__handle = tf.get_session_handle(c__)
#                c__handle = sess.run(c__handle, feed_dict = {target_value:target_values,
#                    x:training_features, s_len:seq_len,
#                    c_h:c_handle.handle,h_h:h_handle.handle})
#                h__handle = tf.get_session_handle(h__)
#                h__handle = sess.run(h__handle, feed_dict = {target_value:target_values,
#                    x:training_features, s_len:seq_len,
#                    c_h:c_handle.handle,h_h:h_handle.handle})
#            else:
#                sess.run(train_step, feed_dict = {target_value:target_values,\
#                    x:training_features, s_len:seq_len,\
#                    c_h:c__handle.handle,h_h:h__handle.handle})
#
#                c__, h__ = state
#                c__handle = tf.get_session_handle(c__)
#                c__handle = sess.run(c__handle, feed_dict = {target_value:target_values,
#                        x:training_features, s_len:seq_len,
#                        c_h:c__handle.handle, h_h:h__handle.handle})
#                h__handle = tf.get_session_handle(h__)
#                h__handle = sess.run(h__handle, feed_dict = {target_value:target_values,
#                        x:training_features, s_len:seq_len,
#                        c_h:c__handle.handle,h_h:h__handle.handle})
#
            if i%50 == 0:
                print "iteration %d"%i
                print "cost %f"%cost.eval(feed_dict = {target_value:tv,\
                    x:tr_fe, s_len:sl})
     #               c_h:c__handle.handle,h_h:h__handle.handle})
                print "tf accuracy %f"%accuracy.eval(feed_dict = {target_value:tv,\
                    x:tr_fe, s_len:sl})
#                c_h:c__handle.handle,h_h:h__handle.handle})

                y = observe.eval(feed_dict = {\
                        x:tr_fe, s_len:sl})
#                    c_h:c__handle.handle,h_h:h__handle.handle})
#        print tv
#        print y
                fpr, tpr, thresholds = metrics.roc_curve(tv, y, pos_label=1)
                auc = metrics.auc(fpr,tpr)
                print "area under curve is %f"%auc
                f1s = metrics.f1_score(tv,y)
                print "f1-score is %f"%f1s
                sk_accuracy = metrics.accuracy_score(tv,y)
                print "accuracy by sklearn %f"%sk_accuracy
        break
