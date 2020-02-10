import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, BatchNormalization, Flatten, Dropout
from keras.models import Model
from keras import backend as K


def load_train_data(filepath):
    print("Loading training data...") 
    # Loading dataset
    data  = pickle.load(open(filepath, 'rb'))
    X_train_orig, Y_train = data['X_train'], data['Y_train']

    # Normalize image vectors
    X_train = X_train_orig.astype('float32')/255 
    return (X_train, Y_train)
    
def load_test_data(filepath):
    print("Loading testing data...")
    # Loading dataset
    data  = pickle.load(open(filepath, 'rb'))
    X_test_orig, Y_test =  data['X_test'], data['Y_test']

    # Normalize image vectors
    X_test = X_test_orig.astype('float32')/255
    
    return (X_test, Y_test)


class Privatizer:
  
    def __init__(self, img_shape):
        self.img_shape = img_shape
        
        # Build discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                  optimizer='adam',
                                  metrics=['accuracy'])
        self.discriminator.summary()

        # Build the generator
        self.channel = self.build_channel()
        self.channel.compile(optimizer = 'adam', loss = 'mean_squared_error')
        
        # Build combined model
        self.discriminator.trainable = False
        self.discriminator.summary()
        
        img_orig = Input(shape = self.img_shape)
        img_priv = self.channel(img_orig)

        priv_info_predict = self.discriminator(img_priv)
        self.combined = Model(input = img_orig, output = [img_priv, priv_info_predict])
        self.combined.compile(loss={'model_2': 'mean_absolute_error', 'model_1':'binary_crossentropy'}, loss_weights = {'model_2':35, 'model_1': -1.0},  optimizer='adam')
        
        return None


    def build_channel(self):

        input_img = Input(shape = self.img_shape)  # adapt this if using `channels_first` image data format

        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        #x = MaxPooling2D((2, 2), padding='same')(x)
        #x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Dense(128, activation='relu')(x)
        encoded = Dense(128, activation='relu')(x)

        # at this point the representation is (4, 4, 8) i.e. 128-dimensional
         
        x = Dense(128, activation='relu')(encoded)
        x = Dense(128, activation='relu')(x)
        #x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        #x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding = 'same')(x)
        x = Conv2D(32, (3, 3), activation='relu', padding = 'same')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name = 'channel')(x)
        
        return Model(input_img, decoded)
      

    def build_discriminator(self):
      
        input_img = Input(shape = self.img_shape)

        x = Conv2D(32, kernel_size=3, strides=2, padding="same")(input_img)
        x = Dropout(0.25)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Dropout(0.25)(x)

        #x = Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
        #x = LeakyReLU(alpha=0.2)(x)
        #x = BatchNormalization(momentum=0.8)(x)
        #x = Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
        #x = LeakyReLU(alpha=0.2)(x)
        #x = BatchNormalization(momentum=0.8)(x)
        #x = MaxPooling2D((2, 2), padding='same')(x)
        #x = Dropout(0.25)(x)
        
        x = Flatten()(x)
        x = Dense(256, activation = 'relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(64, activation = 'relu')(x)
        x = Dropout(0.5)(x)
        validity = Dense(1, activation='sigmoid')(x)

        return Model(input_img, validity)
    
    def train(self, X_train_1, Y_train_1, X_train, Y_train,  X_test, Y_test, epochs=300, batch_size = 128):
       
        # Pre-train discrinimator
        self.discriminator.summary()
        self.discriminator.trainable = True 
        self.discriminator.fit(X_train_1, Y_train_1,
                      epochs = 10,
                      batch_size = 128,
                      shuffle = True,
                      validation_data = (X_test, Y_test))
        # Pre-train channel
        self.channel.summary()
        self.channel.fit(X_train, X_train,
                    epochs = 25,
                    batch_size = 128,
                    shuffle=True,
                    validation_data=(X_test, X_test))
        img_auto = self.channel.predict(X_test[0:100])
        pickle.dump(img_auto, open("auto_v2_imgs", "wb"))

        
        for epoch in range(epochs):
            #--------------------
            # Train discriminator
            #--------------------
            self.discriminator.trainable = True
            for _ in range(14):
                idx = np.random.randint(0, X_train_1.shape[0], batch_size)
                priv_imgs = self.channel.predict(X_train_1[idx])
                d_loss = self.discriminator.train_on_batch(priv_imgs, Y_train_1[idx])
            
            #--------------------
            # Train auto-encoder
            #--------------------
            self.discriminator.trainable = False 
            for _ in range(25):
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                channel_loss = self.combined.train_on_batch(X_train[idx], [X_train[idx], Y_train[idx]])

            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], channel_loss[0]))
            if epoch%50 == 0:
                print(self.channel.predict(X_test[0:1]))
        return
        
      

if __name__ == "__main__":
    
    # To avoid discriminator "memorizes" training samples, we separates training data in two parts

    #(X_train_1, Y_train_1) = load_train_data("./data/celeba_train_1.p")
    (X_train_1, Y_train_1) = load_train_data("./data/celeba_train_2.p")
    (X_train, Y_train) = load_train_data("./data/celeba_train_1.p")
    (X_test, Y_test) = load_test_data("./data/celeba_test.p")

    #X_train = np.concatenate((X_train_1, X_train_2[:10000]))
    #Y_train = np.concatenate((Y_train_1, Y_train_2[:10000]))
    
    # Normalize labels into {0, 1}
    Y_train = (Y_train+1)/2
    Y_test = (Y_test+1)/2
    Y_train_1 = (Y_train_1+1)/2
    
    #Plot data
    #plt.imshow(cv2.cvtColor(X_train[0], cv2.COLOR_BGR2RGB))
    #plt.show()

    print ("number of training examples = " + str(X_train.shape[1]))
    print ("number of test examples = " + str(X_test.shape[1]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))

    print(Y_train[1:10])
    print(Y_test[1:10])
    
    priv = Privatizer(X_train[0].shape)

    priv.train(X_train_1, Y_train_1, X_train, Y_train, X_test, Y_test)
    
    decoded_imgs = priv.channel.predict(X_test[0:100])
    pickle.dump(decoded_imgs, open("rec_celea_imgs", "wb"))
