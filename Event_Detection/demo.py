from keras.layers import Dense,Flatten, Conv3D, MaxPool3D,Input,BatchNormalization , Dropout
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.models import Model
import numpy as np
from keras.utils import np_utils
from model_sports import c3d_model
from gen_frame_list import gen_frame_list
from keras.optimizers import SGD,Adam
import cv2
import os
from keras.utils import multi_gpu_model
from math import floor
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import tensorboard as tensorboard
import random
import matplotlib
import pandas as pd
matplotlib.use('AGG')
#print("Entering with batch_size=32,img_size=150,120 having 4 biased classes+changed model")
config = tf.ConfigProto(device_count={'GPU':1, 'CPU':4})






#-------set_clip_size    and    set_batch_size-------------------------------------#

batch_size=8
no_frame  =16

#-------------------------------------------set_path_trainTest_direstory_Images-----------------------------#

#train_dir  = "/home/student/usama_lahore/Ramna/trained_images/train/"
test_dir=  "/home/student/usama_lahore/Ramna/trained_images/test/"

#class_list_train,train_frame_list = gen_frame_list(train_dir,True)
class_list_test,test_frame_list = gen_frame_list(test_dir,True)
print(np.array(test_frame_list).shape)
print(np.array(class_list_test).shape)
print(len(test_frame_list)//(no_frame * batch_size))

def gen_data(files_list,categories):
    """"Replaces Keras' native ImageDataGenerator."""    
    no_frame = 16
    x_train = []
    y_train = []
    i       = 0
    while i < ((len(files_list)/no_frame)- 10):
#         print("Frame:" , i)
        start = i *  no_frame
        end   = start + (no_frame)                    
        stack_of_16 = []
#         print("START: {}, END: {}".format(start,end))
        for frame in range(start,end):
#             print("Frame No: {}".format(frame))
            image = cv2.imread(files_list[frame])
            image = cv2.resize(image,(170,170))
            image = image / 255.
            #stack_of_16.append(image)       
            
        y_train.append(categories.index(files_list[start].split("/")[7]))        
       # x_train.append(np.array(stack_of_16))
        i = i+1
    return np_utils.to_categorical(y_train,14)
yy_test = gen_data(test_frame_list,class_list_test)
print(yy_test.shape)