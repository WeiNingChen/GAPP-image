import os, sys
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt

SMILE_FOLDER = './genki4k/'
F_SMILE_FOLDER = './data/smile_data/'

def load_data(filepath):
    
    # Loading dataset
    data  = pickle.load(open(filepath, 'rb'))

    # Plot data
    for i in range(15):
        plt.imshow(cv2.cvtColor(data[i+50], cv2.COLOR_BGR2RGB))
        plt.show()
    
    return 

if __name__ == "__main__":
    #print("length of sys.argv:" + str(len(sys.argv)))
    if len(sys.argv) > 1:
        load_data("./rec_celea_imgs_" + sys.argv[1])
    else:    
        load_data("./rec_celea_imgs")
    

