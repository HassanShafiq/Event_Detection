#-----------load)_saved_weights and train--------------------------------#
from keras.models import model_from_json
from keras.layers import Dense,Flatten, Conv3D, MaxPool3D,Input,BatchNormalization , Dropout
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy
from keras.models import Model
import numpy as np
from keras.utils import np_utils
from models import c3d_model
from gen_frame_list import gen_frame_list
from keras.optimizers import SGD,Adam
import cv2
import os
from math import floor
from keras import backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import tensorboard as tensorboard
import random
import matplotlib
matplotlib.use('AGG')
print("Entering........................................................")



#---------------set_clip_size    and    set_batch_size-------------------------------------#

batch_size=8
no_frame  =16

#-------------------------------------------set_path_trainTest_direstory_Images-----------------------------#

train_dir  = r"D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\TrimmedImages\train_imagess_2\\"
test_dir=  r"D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\TrimmedImages\TestImagaes\\"

class_list_train,train_frame_list = gen_frame_list(train_dir,True)
class_list_test,test_frame_list = gen_frame_list(test_dir,True)


def generate_data(files_list,categories , batch_size):
    """"Replaces Keras' native ImageDataGenerator."""    
    if len(files_list) != 0:
#         print("Total Frames: ", len(files_list))
        cpe = 0 
        while True:
            if cpe == floor(len(files_list)/ (batch_size * no_frame)):
                cpe = 0
#             for cpe in range(floor(len(files_list)/ (batch_size * no_frame))):
            x_train = []
            y_train = []
#             print('Cycle No: ', cpe)
            c_start  = batch_size * cpe 
            c_end    = (c_start + batch_size)
#             print("C_Start:",c_start, " c_end: ", c_end)
            for b in range(c_start, c_end):
#                 print('  Frame Set: ',b)
                start = b *  no_frame
                end   = start + (no_frame)                    
                stack_of_16=[]
                for i in range(start,end):                  
                   # print('    Frame Index: ',files_list[i])
                    

                    image = cv2.imread(files_list[i])
                    
                    image = cv2.resize(image,(150,150))
                    image = image / 255
                    stack_of_16.append(image)
                  
      
                    
#                 print("Path : ", files_list[start])
#                 print("Class: ", files_list[start].split("/")[4])
#                 print("Cat Index: ",categories.index(files_list[start].split("\\")[5]))
                y_train.append(categories.index(files_list[start].split("\\")[8]))
                
#                 print("y_train",y_train)
                x_train.append(np.array(stack_of_16))
            cpe += 1
#                 print("y_train",np_utils.to_categorical(y_train,2))

#                 print("x_train",np.array(x_train).shape)
#                 print("y_train",np.array(y_train).shape)
#             print("Total Frames:_x_train ", len(x_train))
            yield(np.array(x_train).transpose(0,1,2,3,4),np_utils.to_categorical(y_train,14))

model = c3d_model()
model.load_weights(r'D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\TrimmedImages\ucf_crime_weights_file.h5')
model.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop', metrics=['accuracy'])

loss,acc=model.evaluate_generator(
    generate_data(test_frame_list,class_list_test,batch_size),steps=floor(len(test_frame_list)/ (batch_size * no_frame)))
print(loss,acc)
 