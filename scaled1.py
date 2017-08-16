#import matplotlib.pyplot as plt
from sklearn import metrics
import tensorflow as tf
from  read_proces_data import *
sess = tf.InteractiveSession()

# create lstm cell in tensorflow
hidden_units = 128
lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units =\
        hidden_units,state_is_tuple=True)

batch_size = 60
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
rb = tf.constant(0.1,shape=[1],dtype=tf.float32)
b = tf.Variable(rb,dtype=tf.float32)
# initialize variables
# calculate final output
logits = tf.matmul(output, W) + b
observed = tf.sigmoid(logits)

# calculate cost as calculated in logistic classification.
logistic_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
        labels = target_value, logits = tf.reshape(logits,[batch_size])))
#cost = tf.reduce_mean(-1.0 * target_value * tf.log(observed) - \
#        (1 - target_value) * tf.log(1 - observed))
#cost = tf.reduce_mean(-1 * target_value * tf.log(observed))
#cost = tf.reduce_mean(tf.square(target_value - observed))
# optimize the cost
train_step = tf.train.AdamOptimizer(1e-4).minimize(logistic_cost)
def threshold_fn(array):
    return array >= 0.50
obser = tf.py_func(threshold_fn,[observed], tf.bool)
observe = tf.cast(obser,tf.float32)

sess.run(tf.global_variables_initializer())

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

        scaled_features = feature_scaling(training_features, batch_size,\
                seq_len)

        for i in range(0,3000):
            tr_fe, sl , tv =\
            sample_data(scaled_features,target_values,seq_len,batch_size)

            sess.run(train_step, feed_dict = {target_value:tv,\
                    x:tr_fe, s_len:sl})

            if i%100 == 0:
                print "iteration %d"%i
                print "cost %f"%logistic_cost.eval(feed_dict = {target_value:tv,\
                    x:tr_fe, s_len:sl})

                y = observe.eval(feed_dict = {\
                        x:tr_fe, s_len:sl})
#        print tv
#        print y
                f1s = metrics.f1_score(tv,y)
                print "f1-score is %f"%f1s
                fpr, tpr, thresholds = metrics.roc_curve(tv, y, pos_label=1)
                auc = metrics.auc(fpr,tpr)
                print "area under curve is %f"%auc
                sk_accuracy = metrics.accuracy_score(tv,y)
                print "accuracy %f"%sk_accuracy
                print " "

                yfile = observe.eval(feed_dict = {\
                        x:scaled_features, s_len:seq_len})

                f1score = metrics.f1_score(target_values,yfile)
                print "f1-score of file is %f"%f1score
                fprr, tprr, thresholds = metrics.roc_curve(target_values, yfile, pos_label=1)
                aucc = metrics.auc(fprr,tprr)
                print "area under curve of file is %f"%aucc
                sk_acc = metrics.accuracy_score(target_values,yfile)
                print "accuracy of file is %f"%sk_acc
                print " "
        break
