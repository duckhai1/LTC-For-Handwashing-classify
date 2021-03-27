import os
import glob
from matplotlib import pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2

from skimage.feature import hog

RAWDATA_PATH = os.path.join("..", "raw_data")
VIDEO_EXTENSION = ('.mp4', '.avi')

PROCESS_VIDEO_LENGTH = 180 # frames

## VIDEO PROPERTY
FPS = 30 #fps

label_map = {
    "Step_1": 0,
    "Step_2_Left": 1,
    "Step_2_Right": 1,
    "Step_3": 2,
    "Step_4_Left": 3,
    "Step_4_Right": 3,
    "Step_5_Left": 4,
    "Step_5_Right": 4,
    "Step_6_Left": 5,
    "Step_6_Right": 5,
    "Step_7_Left": 6,
    "Step_7_Right": 6
}


def extract_video():
    # === for testing
    test = 0
    debug = False
    # ===

    # read each type of dataset (train/test)
    dataset_list = glob.glob(os.path.join(RAWDATA_PATH, "*"))
    for dataset_path in dataset_list:
        date_type = os.path.basename(dataset_path)
        print(date_type)
        # read each label in dataset
        for _, label_list, _ in os.walk(dataset_path):
            for label in label_list:
                print("\t" + label)

                # read each video file in each label
                for dirname, _, filenames in os.walk(os.path.join(dataset_path, label)):
                    for filename in filenames:
                        if filename.endswith(VIDEO_EXTENSION):
                            # === for testing
                            if test > 4 and debug == True:
                                break
                            test += 1 

                            # ===

                            video_path = os.path.join(dirname, filename)
                            print("\t\t" +video_path)

                            ## already preprocess yet: 
                            feature_path = os.path.join("..", "clean_data", date_type, label, filename[:-4]+"_sv_0.npy")
                            if os.path.isfile(feature_path):
                                continue

                            cap = cv2.VideoCapture(video_path)


                            no_frames =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            no_sub_video = no_frames // PROCESS_VIDEO_LENGTH
                            # process each sub video
                            for sub_video_counter in range(no_sub_video):
                                frame_list = []
                                path = os.path.join("..", "clean_data", date_type, label, filename[:-4]+"_sv_"+str(sub_video_counter))
                                print("\t\t  > divide to:" + path)
                                # process each image
                                for img_counter in range(PROCESS_VIDEO_LENGTH):
                                    # read each frame
                                    ret, frame = cap.read() 
                                    
                                    if not ret:
                                        frame_list=[]   # not using this sub_video 
                                        break
                                        
                                    # extract feature from image
                                    ## TODO Adapt the parameter and preprocessing/scale the image
                                    resize_frame = cv2.resize(frame, (120, 180), interpolation = cv2.INTER_AREA)
                                    feature_vector, hog_img = hog(
                                        resize_frame, pixels_per_cell=(14,14), 
                                        cells_per_block=(2, 2), 
                                        orientations=9, 
                                        visualize=True, 
                                        block_norm='L2-Hys',
                                        feature_vector=True)

                                    # adding label (y) in data
                                    feature_vector = np.append(feature_vector, [label_map[label]] )
                                    frame_list.append(feature_vector)
                                np.save(path, frame_list)

                            
if __name__ == '__main__':
    extract_video()                
