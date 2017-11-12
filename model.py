import numpy as np
import cv2
import pandas as pd
import sys
import csv

def data_generator(csv_dataframe, img_dir_path, batch_size):

    while True: 
        #parse the row for image file name and steering angle
        #read image and steering angle into an array
        #yield array after n images have been read in
        X_train = []
        y_train = []
        count = 0

        for row in csv_dataframe.itertuples():    
            
            img_name = row[1].split('/')[-1]
            img = cv2.imread(img_dir_path+"/"+img_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            steer_angle = row[4]

            X_train.append(img)
            y_train.append(steer_angle) 
            count +=1
            
            if count == batch_size:
               yield np.array(X_train), np.array(y_train)
               X_train = []
               y_train = []
               count = 0

    
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

def train_model(log_file_path, img_dir_path):
    
    print ('log file path: ', log_file_path)
    print('img dir path: ', img_dir_path)

    batch_size = 10
    csv_data = pd.read_csv(log_file_path)
    file_generator = data_generator(csv_data, img_dir_path, batch_size)

    x,y = next(file_generator)
    print('x train_shape : ',x.shape)
    print('y train shape: ', y.shape)

    x,y = next(file_generator)
    print('x train_shape : ',x.shape)
    print('y train shape: ', y.shape)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ("usage python model.py <path to log file> <path to image files>") 
        exit()

    train_model(sys.argv[1],sys.argv[2])







