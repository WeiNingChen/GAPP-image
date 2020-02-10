import os, sys
import pickle
import cv2
import numpy as np
from matplotlib import pyplot as plt


def load_data(filepath):
    
    # Loading dataset
    data  = pickle.load(open(filepath, 'rb'))

    # Plot data
    #for i in range(15):
    #    plt.imshow(cv2.cvtColor(data[i+50], cv2.COLOR_BGR2RGB))
    #    plt.show()
    
    return data 

if __name__ == "__main__":
    #print("length of sys.argv:" + str(len(sys.argv)))
    if len(sys.argv) > 1:
        data = load_data("./rec_celea_imgs_" + sys.argv[1])
        if not os.path.exists("./rec_imgs_" + sys.argv[1]):
            os.makedirs("./rec_imgs_"+sys.argv[1])
        for i in range(50):
            img = cv2.resize(data[i]*255, (178, 218))
            cv2.imwrite("./rec_imgs_" + sys.argv[1]+str("/%04d.jpg"%i)+".jpg", img)

    else:    
        print("Please specify imgs!!")
    

