import os
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_data(filepath):
    
    
    # Loading dataset
    data  = pickle.load(open(filepath, 'rb'))
    # Plot data
    for i in range(15):
        plt.imshow(cv2.cvtColor(data[i+50], cv2.COLOR_BGR2RGB))
        plt.show()
    
    return 

if __name__ == "__main__":
    #data = pickle.load(open("data/",'rb'))
    #X_test = data['X_test']
    #pickle.dump(X_test[0:100], open("./ans_imgs",'wb'))
    load_data("./ans_imgs")
    

