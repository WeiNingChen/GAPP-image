import os
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt

IMG_FOLDER = './celeba/img_align_celeba/'
LABEL_FOLDER = './celeba/'
DATA_FOLDER = './data/'
NUM_IMAGE = 110000 
WID_MEAN = 216
LEN_MEAN = 176
#SMILE_SIZE = 48

data = []
labels = []

print("Loading data...")
with open(LABEL_FOLDER+"list_attr_celeba.txt") as f:
    # Skip first two lines
    f.readline()
    f.readline()

    for i in range(NUM_IMAGE):
        fileName = IMG_FOLDER + str("%06d" % (1+i,)) + ".jpg"
        #print(fileName)
        img = cv2.imread(fileName)
        img = cv2.resize(img, (WID_MEAN, LEN_MEAN), cv2.INTER_AREA) 
        #T = np.zeros([SMILE_SIZE, SMILE_SIZE, 1])
        #T[:, :, 0] = img
        l = f.readline()
        label = (int)(l.split()[32])
        data.append(np.array(img))
        labels.append(label)
        if i%5000 == 0:
          print(str(i+1)+" images loaded...")


data_array = np.array(data)
print('data_array shape:')
print(data_array.shape)

labels_array = np.array(labels)
print("label's shape:")
print(labels_array.shape)

print(labels_array[0:20])

indices = np.arange(data_array.shape[0])

print("Shuffling dataset...")
for _ in range(10):
    np.random.shuffle(indices)

data_array = data_array[indices]
labels_array = labels_array[indices]

train_X, test_X = data_array[:90000], data_array[100000:110000]
train_Y, test_Y = labels_array[:90000], labels_array[100000:110000]

data_np_train_1 = {}
data_np_train_2 = {}
data_np_test = {}

# Split training data so that each dataset < 4GB
data_np_train_1['X_train'] = train_X[:35000]
data_np_train_1['Y_train'] = train_Y[:35000]

data_np_train_2['X_train'] = train_X[35000:70000]
data_np_train_2['Y_train'] = train_Y[35000:70000]

data_np_test['X_test'] = test_X
data_np_test['Y_test'] = test_Y

print("Saving dataset...")
#np.save(F_SMILE_FOLDER + 'data.npy', data_np)
pickle.dump( data_np_test, open( DATA_FOLDER + "celeba_test.p", "wb" ) )
pickle.dump( data_np_train_1, open( DATA_FOLDER + "celeba_train_1.p", "wb" ) )
pickle.dump( data_np_train_2, open( DATA_FOLDER + "celeba_train_2.p", "wb" ) )
