import numpy as np
import cv2
import pandas as pd
import sys
import csv
from sklearn.model_selection import train_test_split

def data_generator(csv_dataframe, img_dir_path, batch_size,validation = True):

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
from keras.layers.core import Dense, Activation, Flatten,Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D



def build_linear_model(img_size = (160,320,3)):
    model = Sequential()
    model.add(Lambda(lambda x:x/255.-0.5,input_shape=img_size))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(32))
    model.add(Dense(1))
    model.compile(loss='mse',optimizer='adam')

    return model


def build_lenet_model(img_size=(160,320,3)):

    lenet_model = Sequential()
    lenet_model.add(Lambda(lambda x:x/255. - 0.5 , input_shape=img_size))
    lenet_model.add(Convolution2D(6,(5,5),padding = 'valid',activation='relu'))
    lenet_model.add(MaxPooling2D())
    lenet_model.add(Convolution2D(16,(5,5),padding='valid',activation='relu'))
    lenet_model.add(MaxPooling2D())
    lenet_model.add(Flatten())
    lenet_model.add(Dense(128))
    lenet_model.add(Dense(84))
    lenet_model.add(Dense(1))
    lenet_model.compile(loss ='mse', optimizer='adam')

    return lenet_model

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

def train_model(model, train_batch_generator,
        valid_data_generator,train_steps_per_epoch=4,
        valid_steps_per_epoch =2,num_epochs=5):
    
    model.fit_generator(train_batch_generator,
            steps_per_epoch=train_steps_per_epoch,
            epochs=5,verbose = 1,
            validation_data=valid_data_generator,
            validation_steps=valid_steps_per_epoch,
            shuffle=True)

    return model

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

def load_data(log_file_path,test_ratio = 0.3):
    csv_data = pd.read_csv(log_file_path)
    y_dummy_data = csv_data.iloc[:, 0]
    X_train, X_valid, y_train, y_valid = train_test_split(csv_data, y_dummy_data, test_size = test_ratio)

    return X_train,y_train, X_valid, y_valid


def behavioral_cloning_pipeline(log_file_path, img_dir_path):
    print ('log file path: ', log_file_path)
    print('img dir path: ', img_dir_path)
    X_train, y_train, X_valid, y_valid = load_data(log_file_path,test_ratio = 0.0)
    print('x train shape', X_train.shape)
    print('x valid shape', X_valid.shape)
    batch_size = 10
    train_data_generator = data_generator(X_train, img_dir_path, 
            batch_size=batch_size, validation = False)
    valid_data_generator = data_generator(X_valid, img_dir_path, 
            batch_size = batch_size, validation = True)

    #model = build_model()
    model = build_lenet_model(img_size=(160,320,3))
    #model = train_model(model,train_data_generator,valid_data_generator,
    #        train_steps_per_epoch=3,valid_steps_per_epoch=1)
    
    model = train_to_overfit_model(model, X_train,
            img_dir_path,data_size=3,n_epochs=5)
    test_model(model,test_data_df,img_dir_path)


def test_pipeline(log_file_path, img_dir_path):
    print ('log file path: ', log_file_path)
    print('img dir path: ', img_dir_path)
    model = build_lenet_model(img_size=(160,320,3))
    X_train = pd.read_csv(log_file_path,header=None) 
    print("train shape: ", X_train.shape)
    model = train_to_overfit_model(model, X_train,
            img_dir_path,data_size=3,n_epochs=5)
    test_model(model,X_train,img_dir_path)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ("usage python model.py <path to log file> <path to image files>") 
        exit()
    test_pipeline(sys.argv[1],sys.argv[2])
    #behavioral_cloning_pipeline(sys.argv[1],sys.argv[2])







