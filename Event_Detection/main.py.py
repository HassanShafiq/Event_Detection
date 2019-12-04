from keras.layers import Dense,Flatten, Conv3D, MaxPool3D,Input,BatchNormalization , Dropout
from keras.optimizers import Adadelta
from keras.optimizers import SGD,adam, Adagrad
from keras.losses import categorical_crossentropy
from keras.models import Model
import numpy as np
from keras.utils import np_utils
from model_sports import c3d_model
from gen_frame_list import gen_frame_list
from keras.optimizers import SGD,Adam
from keras.callbacks import ModelCheckpoint
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
from sklearn.model_selection import train_test_split

matplotlib.use('AGG')
#print("Entering with batch_size=32,img_size=150,120 having 4 biased classes+changed model")
config = tf.ConfigProto(device_count={'GPU':0, 'CPU':4})






#-------set_clip_size    and    set_batch_size-------------------------------------#

batch_size=16
no_frame  =16

#-------------------------------------------set_path_trainTest_direstory_Images-----------------------------#

train_dir  = "/train/"
test_dir=  "/test/"

class_list_train,train_frame_list = gen_frame_list(train_dir,True)
class_list_test,test_frame_list = gen_frame_list(test_dir,True)
print(np.array(train_frame_list).shape)
print(np.array(class_list_train).shape)
print(np.array(test_frame_list).shape)
print(np.array(class_list_test).shape)
print(len(test_frame_list)//(no_frame * batch_size))
#--------------------------generate DATA--------------------------------------------------
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
#                     print('    Frame Index: ',i)
                    image = cv2.imread(files_list[i])
                    image = cv2.resize(image,(170,170))
                    image = image / 255.
                    stack_of_16.append(image)       
                    
                 #   print("Path : ", files_list[i])
#                 print("Class: ", files_list[start].split("/")[4])
#                 print("Cat Index: ",categories.index(files_list[start].split("/")[4]))
                y_train.append(categories.index(files_list[start].split("/")[7]))
              #  print("y_train",y_train)
                x_train.append(np.array(stack_of_16))
            cpe += 1
#                 print("y_train",np_utils.to_categorical(y_train,2))

#                 print("x_train",np.array(x_train).shape)
#                 print("y_train",np.array(y_train).shape)
#             print("Total Frames:_x_train ", len(x_train))
            yield(np.array(x_train).transpose(0,1,2,3,4),np_utils.to_categorical(y_train,12))

#----------train on multiple gpus---------------------------------------#

model1=c3d_model()
# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
from keras.models import model_from_json

#model = model_from_json(open('/home/student/usama_lahore/Ramna/Results/6-epochs-29%acc-overfit/ucf_crime.json', 'r').read())
#model.summary()
#model=c3d_model()
#model.load_weights('/home/student/usama_lahore/Ramna/Results/6-epochs-29%acc-overfit/ucf_crime_weights_file.h5')
model = multi_gpu_model(model1, gpus=2)
le=0.001
opt = SGD(lr=le, momentum=0.009, decay=le)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])
#train_frame_list1,class_list_train1,test_frame_list1,class_list_test1= train_test_split(generate_data(train_frame_list,class_list_train,batch_size),test_size=0.33, random_state=42)

#----------------------------Strat_Training-------------------------------------------#
#filepath = '/home/student/usama_lahore/Ramna/trained_images/train/ucf_crime_weights_file.h5'
#checkpoint = ModelCheckpoint(filepath, monitor='val_accuraccy', verbose=1, save_best_only=True, mode = 'max')
#callbacks_list = [checkpoint]
model.fit_generator(generate_data(train_frame_list,class_list_train,batch_size), 
                    steps_per_epoch=floor(len(train_frame_list)/(no_frame * batch_size)),
                    epochs=50,
                     validation_data=generate_data(test_frame_list,class_list_test,batch_size),
                     validation_steps=floor(len(test_frame_list)/ (no_frame * batch_size )),
                     verbose=1)
#model1.save_weights("/home/student/usama_lahore/Ramna/code_files/ucf_crime_model_weights_05.h5")              

#----------------------evaluate Model----------------------#
model_json=model.to_json()
with open('/home/student/usama_lahore/Ramna/trained_images/train/ucf_crime1.json',"w") as json_file:
    json_file.write(model_json)
model.save_weights('/home/student/usama_lahore/Ramna/trained_images/train/ucf_crime_weights_file2.h5')

model.save('/home/student/usama_lahore/Ramna/trained_images/train/ucf_model.h5')

                  #-----------------------save_model_file----------------------------------------#
#loss,acc = model.evaluate_generator(
 #   generate_data(train_frame_list,class_list_train,batch_size),steps=floor(len(train_frame_list)/ (batch_size * no_frame)))
#print("loss_train_data",loss)
#print("accuracy_training",acc)
loss,acc = model.evaluate_generator(
    generate_data(test_frame_list,class_list_test,batch_size),steps=floor(len(test_frame_list)/(batch_size * no_frame)))
print("loss_test_data",loss)
print("accuracy_test",acc)
loss,acc = model.evaluate_generator(
    generate_data(train_frame_list,class_list_train,batch_size),steps=floor(len(train_frame_list)/ (batch_size * no_frame)))
print("loss_train_data",loss)
print("accuracy_training",acc)

#model1.save(r"D:\Ramna work\Trim_Dataset\New folder (2)\ucf_crime_model.h5")
