import numpy as np
import os
import csv
from sklearn import preprocessing

# read train data file names
file_path = "BCI_chal_2015_data/train/"
files = os.listdir(file_path)
file_dir = sorted(files)

# read target values from csv file into python list
def get_target_values(file_name):
    target_values_file = "BCI_chal_2015_data/TrainLabels.csv"
    target_val = []
    with open(target_values_file) as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if file_name[5:-4] in row[0]:
                target_val.append(row[1])
    # extract second column as int8 values
    target_values = np.asarray(target_val,dtype=np.int8)
    return target_values

# read train data from 1 file at a time
def get_train_data(file_name):
    # create full file path 
    full_file_path = file_path + file_name
    #print full_file_path

    # read data from file, data is available from row index 0
    data = np.genfromtxt(full_file_path, delimiter=',')
    return data[1:,:]

# preprocess data points so that all realted inputs upto feedback events can be
# accessed just by one index of array.
def prepare_data(features):
    # first reject uncessary features
    updated_feat = features[:,1:-1]
    number_of_columns = updated_feat.shape[1]
    # find index of 1's in feedback event feature
    ones_index = np.nonzero(features[:,-1])
    ones_index = np.asarray(ones_index)
    # find difference of ones_index'es and divide by sampling frequency to
    # obtain the time difference between each feedback event
    sampling_frequency = 200
    event_diff_in_sec = np.diff(ones_index) / (sampling_frequency * 1.0)
    # now take mean of every 4 points, since 1 word is of 5 letters and we have
    # taken the difference.
    avg_dif_sec = np.zeros(0)
    four = 4
    for index in range(0, event_diff_in_sec.shape[1], four):
        avg_dif_sec = np.append(avg_dif_sec, np.mean(event_diff_in_sec[0,index:index+four]))
    # repeat the elements 5 times
    avg_dif = np.repeat(avg_dif_sec, four)
    # start collecting data points
    data_points = []
    # consider the 0th index case separately
    indices = range(int(ones_index[0][0] -\
            avg_dif[0]*sampling_frequency),ones_index[0][0]+1)
    data_points.append(updated_feat[indices,:])
    rows_per_2darray = []
    rows_per_2darray.append(len(indices))
    # remaining casea are following
    for index in range(1, ones_index.shape[1]):
        no_of_samples = min(ones_index[0][index] - ones_index[0][index - 1],
                avg_dif[index] * sampling_frequency)
        indices = range(ones_index[0][index]-int(no_of_samples), ones_index[0][index]+1)
        rows_per_2darray.append(len(indices))
        data_points.append(updated_feat[indices,:])

    max_rows = max(rows_per_2darray)
    DATA = np.zeros((ones_index.shape[1], max_rows,number_of_columns))
    for i in range(ones_index.shape[1]):
        temp_data = np.asarray(data_points[i])
        DATA[i,:temp_data.shape[0],:] = temp_data
    rows_per_2darray[:] = [x-1 for x in rows_per_2darray]

#    rows_per_2darray = [rows_per_2darray[i] for i in indices]
    return (DATA, rows_per_2darray)

def sample_data(x,y,seq_len,b_size):
    b_size = b_size/2
    s_len = np.array(seq_len)

    zero_indices = [i for i,e in enumerate(y) if e==0]
    x0 = x[zero_indices,:,:]
    s0 = s_len[zero_indices]
    y0 = y[zero_indices]

    one_indices = [i for i,e in enumerate(y) if e==1]
    x1 = x[one_indices,:,:]
    s1 = s_len[one_indices]
    y1 = y[one_indices]

    zero_len = len(zero_indices)
    indice_0 = np.random.randint(0,zero_len,b_size)
    x0_sample = x0[indice_0,:,:]
    s0_sample = s0[indice_0]
    y0_sample = y0[indice_0]


    one_len = len(one_indices)
    indice_1 = np.random.randint(0,one_len,b_size)
    x1_sample = x1[indice_1,:,:]
    s1_sample = s1[indice_1]
    y1_sample = y1[indice_1]

    x_new = np.concatenate((x0_sample,x1_sample))
    s_new = np.concatenate((s0_sample,s1_sample))
    y_new = np.concatenate((y0_sample,y1_sample))

    xnew_len = x_new.shape[0]
    permu_indices = np.random.permutation(xnew_len)
    x_permu = x_new[permu_indices,:,:]
    s_permu = s_new[permu_indices]
    y_permu = y_new[permu_indices]

    return (x_permu,s_permu,y_permu)

def feature_scaling(features, batch_size, seq_len):
    X_scaled = np.zeros(features.shape)
    for index in range(0,batch_size):
        X_scaled[index,:seq_len[index],:] = \
        preprocessing.scale(features[index,:seq_len[index],:])
    return X_scaled
