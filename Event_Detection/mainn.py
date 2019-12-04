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
from keras.utils import multi_gpu_model
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
def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()





#---------------set_clip_size    and    set_batch_size-------------------------------------#

batch_size=32
no_frame  =16

#-------------------------------------------set_path_trainTest_direstory_Images-----------------------------#

train_dir  = r"D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\del\train\\"
test_dir=  r"D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\del\test\\"

class_list_train,train_frame_list = gen_frame_list(train_dir,True)
class_list_test,test_frame_list = gen_frame_list(test_dir,True)
print(np.array(train_frame_list).shape)
print(np.array(class_list_train).shape)
print(np.array(test_frame_list).shape)
print(np.array(class_list_test).shape)
print(len(train_frame_list)//(no_frame * batch_size))
#--------------------------generate DATA--------------------------------------------------#
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
                    image = cv2.resize(image,(112,112))
                    image = image / 255
                    stack_of_16.append(image)       
                    
#                 print("Path : ", files_list[start])
#                 print("Class: ", files_list[start].split("/")[4])
#                 print("Cat Index: ",categories.index(files_list[start].split("/")[4]))
                y_train.append(categories.index(files_list[start].split("/")[4]))
                
#                 print("y_train",y_train)
                x_train.append(np.array(stack_of_16))
            cpe += 1
#                 print("y_train",np_utils.to_categorical(y_train,2))

#                 print("x_train",np.array(x_train).shape)
#                 print("y_train",np.array(y_train).shape)
#             print("Total Frames:_x_train ", len(x_train))
            yield(np.array(x_train).transpose(0,1,2,3,4),np_utils.to_categorical(y_train,4))

#----------train on multiple gpus---------------------------------------#

model1 = c3d_model()
model1.compile(loss='categorical_crossentropy',
                       optimizer='rmsprop',accuracy=["accuracy"])
# Replicates `model` on 8 GPUs.
# This assumes that your machine has 8 available GPUs.
#model = multi_gpu_model(model1, gpus=2)





#----------------------------Strat_Training-------------------------------------------#

H = model1.fit_generator(generate_data(train_frame_list,class_list_train,batch_size), 
                    steps_per_epoch=floor(len(train_frame_list)/ (batch_size * no_frame)),
                    epochs=1,
                     validation_data=generate_data(test_frame_list,class_list_test,batch_size),
                     validation_steps=floor(len(test_frame_list)/ (batch_size * no_frame)),verbose=1
                    )

#----------------------evaluate Model----------------------#
loss,acc = model1.evaluate_generator(
    generate_data(test_frame_list,class_list_test,batch_size),steps=floor(len(test_frame_list)/ (batch_size * no_frame)))
print(loss,acc)
                  #-----------------------save_model_file----------------------------------------#

model_json=model.to_json()
with open(r'D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\TrimmedImages\ucf_crime.json',"w") as json_file:
    json_file.write(model_json)
model1.save_weights(r'D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\TrimmedImages\ucf_crime_weights_file.h5')

model1.save(r'D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\TrimmedImages\ucf_model.h5')

                #------------------save_history_graphs-------------------------------------------------------#

if not os.path.exists(r'D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\TrimmedImages\results_40\\'):
        os.mkdir(r'D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\TrimmedImages\TrainImgaes\results_40\\')
plot_history(H, r'D:\Ramna work\Trim_Dataset\New folder (2)\UCF_CRIME\TrimmedImages\TrainImgaes\results_40\\')
