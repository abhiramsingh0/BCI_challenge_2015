from sklearn import metrics
import tensorflow as tf
from  read_proces_data import *

# find max no of times to roll back
max_seq_len = find_max_seq_len()
print max_seq_len
# number of times to iterate over whole training dataset
#iter_over_files =  1
#for itera in range(0, iter_over_files):
#    for file_ in file_dir:
#        print file_
#        # get features from current indexed file
#        features = get_train_data(file_)
#        # get target values corresponding to this file
#        target_values = get_target_values(file_)
#        # training features is np array with max size and padding
#        training_features, seq_len = prepare_data(features, 0)#max_seq_len)
#        print training_features.shape[1]
#        print max(seq_len)

