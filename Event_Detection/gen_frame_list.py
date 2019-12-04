import cv2
import natsort
import os
import numpy as np

no_frame=16
def gen_frame_list(directory,shuffle=False):    
    cat        = []
    label      = 0
    files_list = []
#     print(cat)
    for c in natsort.natsorted(os.listdir(directory)):
#         print("Category:",c)
        cat.append(c)
        for f in natsort.natsorted(os.listdir(os.path.join(directory,c))):
            ff = os.path.join(directory,c,f)
            sorted_file_list = natsort.natsorted(os.listdir(ff))
            limit = len(sorted_file_list) - (len(sorted_file_list) % no_frame)            
            for fr in range(limit):                
                files_list.append(os.path.join(ff,sorted_file_list[fr]))     
    if shuffle:
        np_array = np.array(np.array_split(files_list, len(files_list) / no_frame))
        np.random.shuffle(np_array)
        files_list = np_array.flatten().tolist()        
    return cat, files_list                    