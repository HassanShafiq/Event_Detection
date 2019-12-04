import cv2
import natsort
import os
import numpy as np

no_frame=16
def gen_frame_list(directory):
    label      = 0
    files_list = []
#     print(cat)
    for c in natsort.natsorted(os.listdir(directory)):
#         print("Category:",c)
        for f in natsort.natsorted(os.listdir(os.path.join(directory,c))):
            ff = os.path.join(directory,c,f)
            sorted_file_list = natsort.natsorted(os.listdir(ff))
            limit = len(sorted_file_list) - (len(sorted_file_list) % no_frame)
            for fr in range(limit):
                files_list.append(os.path.join(ff,sorted_file_list[fr]))
    return files_list
vid_path =  "/home/student/usama_lahore/Ramna/trained_images/train/"
frame_list = gen_frame_list(vid_path);

for  f in  frame_list:
   image = cv2.imread(f)
   if image is None: 
     print(f)
