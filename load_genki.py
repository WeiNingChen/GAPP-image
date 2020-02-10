import os
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt

SMILE_FOLDER = './genki4k/'
F_SMILE_FOLDER = './data/smile_data/'
NUM_SMILE_IMAGE = 4000
WID_MEAN = 198
LEN_MEAN = 184
#SMILE_SIZE = 48

data = []
labels = []

print("Loading data...")
with open(SMILE_FOLDER + "labels.txt") as f:
    for i in range(NUM_SMILE_IMAGE):
        fileName = SMILE_FOLDER + "files/file" + str("%04d" % (1+i,)) + ".jpg"
        img = cv2.imread(fileName)
        img = cv2.resize(img, (WID_MEAN, LEN_MEAN), cv2.INTER_AREA) 
        #T = np.zeros([SMILE_SIZE, SMILE_SIZE, 1])
        #T[:, :, 0] = img
        l = f.readline()
        label = (int)(l.split()[0])
        data.append(np.array(img))
        labels.append(label)

data_array = np.array(data)
print('data_array shape:')
print(data_array.shape)

labels_array = np.array(labels)
print("label's shape:")
print(labels_array.shape)
indices = np.arange(data_array.shape[0])

print("Shuffling dataset...")
#for _ in range(10):
#    np.random.shuffle(indices)

data_array = data_array[indices]
labels_array = labels_array[indices]

train_X, test_X = data_array[:3000], data_array[3000:]
train_Y, test_Y = labels_array[:3000], labels_array[3000:]

data_np = {}
data_np['X_train'] = train_X
data_np['X_test'] = test_X
data_np['X_train'] = train_Y
data_np['X_test'] = test_Y

print("Saving dataset...")
#np.save(F_SMILE_FOLDER + 'data.npy', data_np)
pickle.dump( data_np, open( F_SMILE_FOLDER + "data.p", "wb" ) )
