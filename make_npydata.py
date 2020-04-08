import numpy as np
import os

shanghaiAtrain_path='/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_A_final/train_data/images/'
shanghaiAtest_path='/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_A_final/test_data/images/'

train_list = []
for filename in os.listdir(shanghaiAtrain_path):
    if filename.split('.')[1] == 'jpg':
        train_list.append(shanghaiAtrain_path+filename)
train_list.sort()
np.save('./ShanghaiA_train.npy', train_list)
print(len(train_list))

test_list = []
for filename in os.listdir(shanghaiAtest_path):
    if filename.split('.')[1] == 'jpg':
        test_list.append(shanghaiAtest_path+filename)
test_list.sort()
np.save('./ShanghaiA_test.npy', test_list)
print(len(test_list))


shanghaiBtrain_path='/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_B_final/train_data/images/'
shanghaiBtest_path='/data/weixu/ShanghaiTech_Crowd_Counting_Dataset_baseline/part_B_final/test_data/images/'

train_list = []
for filename in os.listdir(shanghaiBtrain_path):
    if filename.split('.')[1] == 'jpg':
        train_list.append(shanghaiBtrain_path+filename)
train_list.sort()
np.save('./ShanghaiB_train.npy', train_list)
print(len(train_list))

test_list = []
for filename in os.listdir(shanghaiBtest_path):
    if filename.split('.')[1] == 'jpg':
        test_list.append(shanghaiBtest_path+filename)
test_list.sort()
np.save('./ShanghaiB_test.npy', test_list)
print(len(test_list))


Qnrf_train_path='/data/weixu/UCF-QNRF_ECCV18/dataset_1920/train/images/'
Qnrf_test_path='/data/weixu/UCF-QNRF_ECCV18/dataset_1920/test/images/'

train_list = []
for filename in os.listdir(Qnrf_train_path):
    if filename.split('.')[1] == 'jpg':
        train_list.append(Qnrf_train_path+filename)
train_list.sort()
np.save('./Qnrf_train.npy', train_list)
print(len(train_list))

test_list = []
for filename in os.listdir(Qnrf_test_path):
    if filename.split('.')[1] == 'jpg':
        test_list.append(Qnrf_test_path+filename)
test_list.sort()
np.save('./Qnrf_test.npy', test_list)
print(len(test_list))