import numpy as np
import cv2
import pandas as pd
import sys
import csv
from sklearn.model_selection import train_test_split


# A generator method to load image data in batches
def data_generator(csv_dataframe, img_dir_path, batch_size,
        left_steer_bias = 0.2,
        right_steer_bias = 0.1,
        validation = False):

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
            
            if not validation:
                temp_img = cv2.flip(img,flipCode=1)
                X_train.append(temp_img)
                y_train.append(-1*steer_angle)

                img_name = row[2].split('/')[-1]
                img = cv2.imread(img_dir_path+"/"+img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                X_train.append(img)
                y_train.append(steer_angle-(left_steer_bias*steer_angle))
                
                img_name = row[3].split('/')[-1]
                img = cv2.imread(img_dir_path+"/"+img_name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                X_train.append(img)
                y_train.append(steer_angle + (right_steer_bias*steer_angle))

            count +=1
            
            if count == batch_size:
               yield np.array(X_train), np.array(y_train)
               X_train = []
               y_train = []
               count = 0

    
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten,Lambda, Dropout
from keras.layers.convolutional import Convolution2D,Cropping2D
from keras.layers.pooling import MaxPooling2D



# A simple linear NN model
def build_linear_model(img_size = (160,320,3)):
    model = Sequential()
    model.add(Lambda(lambda x:x/255.-0.5,input_shape=img_size))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')

    return model

# builds a deep CNN using the LeNet architecture
def build_lenet_model(img_size=(160,320,3)):

    lenet_model = Sequential()
    lenet_model.add(Lambda(lambda x:x/255. - 0.5 , input_shape=img_size))
    lenet_model.add(Cropping2D(cropping=((50,20),(0,0))))    
    lenet_model.add(Convolution2D(6,(5,5),padding = 'valid',activation='relu'))
    lenet_model.add(Dropout(rate = 0.4))
    lenet_model.add(MaxPooling2D())
    lenet_model.add(Convolution2D(16,(5,5),padding='valid',activation='relu'))
    lenet_model.add(Dropout(rate = 0.4))
    lenet_model.add(MaxPooling2D())
    lenet_model.add(Flatten())
    lenet_model.add(Dense(128))
    lenet_model.add(Dense(84))
    lenet_model.add(Dense(1))
    lenet_model.compile(loss ='mse', optimizer='adam')

    return lenet_model


# builds a deep CNN model that has architecture similar to that published by Nvidia for end to end driving
def build_nvidia_model(img_size=(160,320,3)):
    #W_out = [(W-F+2P)/2S]+1
    #H_out = [(H-F +2P)/2S]+1

    nvidia_model = Sequential()
    nvidia_model.add(Lambda(lambda x:x/255. - 0.5 , input_shape=img_size))
    nvidia_model.add(Cropping2D(cropping =((65,20),(0,0))))
    nvidia_model.add(Convolution2D(24,(5,5),strides=(2,2),activation='relu'))
    nvidia_model.add(Convolution2D(36,(5,5),strides=(2,2),activation='relu'))
    nvidia_model.add(Convolution2D(48,(5,5),strides=(2,2),activation='relu'))
    nvidia_model.add(Dropout(rate = 0.4))
    nvidia_model.add(Convolution2D(64,(3,3),activation='relu'))
    nvidia_model.add(Convolution2D(64,(3,3),activation='relu'))
    nvidia_model.add(Dropout(rate = 0.4))
    nvidia_model.add(Flatten())
    nvidia_model.add(Dense(100))
    nvidia_model.add(Dense(50))
    nvidia_model.add(Dense(10))
    nvidia_model.add(Dense(1))
    nvidia_model.compile(loss='mse', optimizer='adam')
    return nvidia_model


# trains a model on a small chunk of training data as an initial sanity check
def train_to_overfit_model(model,train_img_df,
        train_img_dir,data_size=3,
        n_epochs=5):
    
    X_train=[]
    y_train=[]
    count = 0

    for row in train_img_df.itertuples():    
        
        img_name = row[1].split('/')[-1]
        img = cv2.imread(train_img_dir+"/"+img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        steer_angle = row[4]

        X_train.append(img)
        y_train.append(steer_angle) 

    img_arr = np.array(X_train)
    y_actual = np.array(y_train) 
    model.fit(img_arr,y_actual,epochs=n_epochs,verbose=1)

    return model


# fits a model to given data
def train_model(model, train_batch_generator,
        valid_data_generator,train_steps_per_epoch=4,
        valid_steps_per_epoch =2,num_epochs=5):
    
    model.fit_generator(train_batch_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=num_epochs,verbose = 1,
            validation_data=valid_data_generator,
            validation_steps=valid_steps_per_epoch,
            shuffle=True)

    return model



# Runs predictions using a provided model 
def test_model(model,test_img_df,test_img_dir,data_size=3):
    
    X_test=[]
    y_test=[]
    count = 0
    for row in test_img_df.itertuples():    
        
        img_name = row[1].split('/')[-1]
        img = cv2.imread(test_img_dir+"/"+img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        steer_angle = row[4]

        X_test.append(img)
        y_test.append(steer_angle) 
        count +=1
        
        if count == data_size:
            break
    img_arr = np.array(X_test)
    y_actual = np.array(y_test) 
    predictions = model.predict(img_arr)
    print('model test results')
    print('predictions: ', predictions)
    print('actual: ',y_test)



## Loads a driving log csv file  and splits into test train datasets
def load_data(log_file_path,test_ratio = 0.3):
    csv_data = pd.read_csv(log_file_path,header=None)
    y_dummy_data = csv_data.iloc[:, 0]
    X_train, X_valid, y_train, y_valid = train_test_split(csv_data, y_dummy_data, test_size = test_ratio)

    return X_train,y_train, X_valid, y_valid


def behavioral_cloning_pipeline(log_file_path, img_dir_path):
    
    # load and print data set statistics
    print ('log file path: ', log_file_path)
    print('img dir path: ', img_dir_path)
    X_train, y_train, X_valid, y_valid = load_data(log_file_path,test_ratio = 0.3)
    print('x train shape', X_train.shape)
    print('x valid shape', X_valid.shape)
    batch_size = 100


    # define a generator to load training data in batches
    train_data_generator = data_generator(X_train, img_dir_path, 
            batch_size=batch_size, 
            left_steer_bias = 0.3,
            right_steer_bias = 0.5, 
            validation = False)
    
    # define data generator to load validata data batches
    valid_data_generator = data_generator(X_valid, img_dir_path, 
            batch_size = batch_size, validation = True)

    # train different types of NN architectures
    #model = build_model()
    #model = build_lenet_model(img_size=(160,320,3))
    model = build_nvidia_model(img_size=(160,320,3))

    #start training  process with validation
    model = train_model(model,train_data_generator,valid_data_generator,
            train_steps_per_epoch=int(X_train.shape[0]/batch_size),
            valid_steps_per_epoch=int(X_valid.shape[0]/batch_size), num_epochs=5)
   
    # train on entire dataset without validation
    #model = train_model(model,train_data_generator,None,
    #                                  train_steps_per_epoch=int(X_train.shape[0]/batch_size),
    #                                  valid_steps_per_epoch=None, num_epochs=10)

    
    # save model to h5 file
    model.save('model.h5')

# runs the whole train and test pipeline on a small chunk of dataset
def test_pipeline(log_file_path, img_dir_path):
   
    print ('log file path: ', log_file_path)
    print('img dir path: ', img_dir_path)
    model = build_lenet_model(img_size=(160,320,3))
    X_train = pd.read_csv(log_file_path,header=None) 
    print("train shape: ", X_train.shape)
    
    #train on a smaller set of data as a sanity check
    test_data_df = X_train.iloc[:11,:]
    model = train_to_overfit_model(model, test_data_df,
            img_dir_path,data_size=10,n_epochs=5)
   
    #run predictions on a small set of data
    test_model(model,test_data_df,img_dir_path, data_size=10)
    
    


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ("usage python model.py <path to log file> <path to image files>") 
        exit()
    #test_pipeline(sys.argv[1],sys.argv[2])
    behavioral_cloning_pipeline(sys.argv[1],sys.argv[2])







