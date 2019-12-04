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
config = tf.ConfigProto(device_count={'GPU':0, 'CPU':4})






#-------set_clip_size    and    set_batch_size-------------------------------------#

batch_size=16
no_frame  =16

#-------------------------------------------set_path_trainTest_direstory_Images-----------------------------#

#train_dir  = "/home/student/usama_lahore/Ramna/trained_images/train/"
test_dir=  "/home/student/usama_lahore/Ramna/new_experiment/test/"


#class_list_train,train_frame_list = gen_frame_list(train_dir,True)
class_list_test,test_frame_list = gen_frame_list(test_dir,True)
print(np.array(test_frame_list).shape)
print(np.array(class_list_test).shape)
print(len(test_frame_list)//(no_frame * batch_size))
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
                    image = cv2.resize(image,(170,170))
                    image = image / 255.
                    stack_of_16.append(image)       
                    
                 #   print("Path : ", files_list[i])
#                 print("Class: ", files_list[start].split("/")[4])
#                 print("Cat Index: ",categories.index(files_list[start].split("/")[4]))
                y_train.append(categories.index(files_list[start].split("/")[7]))
                np.save("/home/student/usama_lahore/Ramna/Results/Results_overfit_41acc/y_train.npy",y_train)
              #  print("y_train",y_train)
                x_train.append(np.array(stack_of_16))
            cpe += 1
#                 print("y_train",np_utils.to_categorical(y_train,2))

#                 print("x_train",np.array(x_train).shape)
           # print("y_train",np.array(y_train).shape)
           # np.save("/home/student/usama_lahore/Ramna/Results/Results_overfit_41acc/",y_train)
#             print("Total Frames:_x_train ", len(x_train))
            yield(np.array(x_train).transpose(0,1,2,3,4),np_utils.to_categorical(y_train,14))

#----------train on multiple gpus---------------------------------------#
from keras.models import model_from_json

#model = model_from_json(open('/home/student/usama_lahore/Ramna/Results/6-epochs-29%acc-overfit/ucf_crime.json', 'r').read())
model=c3d_model()
#model = multi_gpu_model(model, gpus=2)
model.load_weights('/home/student/usama_lahore/Ramna/Results/Results_overfit_41acc/ucf_crime_weights_file.h5')
model.summary()
le=0.001
opt = SGD(lr=le, momentum=0.009, decay=le)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['categorical_accuracy'])


#loss,acc = model.evaluate_generator(
 #           generate_data(test_frame_list,class_list_test,batch_size),
  #          steps=floor(len(test_frame_list)/ (batch_size * no_frame)))
#print("loss",loss)
#print("acc",acc)
y_pred = model.predict_generator(
            generate_data(test_frame_list,class_list_test,batch_size),
            steps=floor(len(test_frame_list)/ (batch_size * no_frame)))
print("y_predict",y_pred.shape)
def gen_data(files_list,categories):
    """"Replaces Keras' native ImageDataGenerator."""    
    no_frame = 16
    x_train = []
    y_train = []
    i       = 0
    while i < ((len(files_list)/no_frame)-10):
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
print("y_test",yy_test.shape)
pred = y_pred.argmax(axis=1)
y_true = yy_test.argmax(axis=1)
from sklearn.metrics import classification_report, confusion_matrix
matrix = confusion_matrix(y_true, pred)
report = classification_report(yy_test.argmax(axis=1), y_pred.argmax(axis=1), target_names=class_list_test)
print("classification_report",report)
import numpy as np


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
#%matplotlib inline 
plot_confusion_matrix(cm           = matrix, 
                      normalize    = False,
                      target_names = class_list_test,
                      title        = "Confusion Matrix")



