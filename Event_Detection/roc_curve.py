from sklearn.metrics import roc_curve, auc
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
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
    while i < ((len(files_list)/no_frame)- 4):
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
y_test = gen_data(test_frame_list,class_list_test)
print(y_test.shape)
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
y_score = model.predict_generator(
            generate_data(test_frame_list,class_list_test,batch_size),
            steps=floor(len(test_frame_list)/ (batch_size * no_frame)))
print("y_predict",y_score.shape)



# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()