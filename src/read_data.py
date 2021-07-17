import os
import glob
import argparse

from matplotlib import pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
from skimage.feature import hog

## import config property
from property import *

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


def extract_train_video():
    # read each label video
    for _, label_list, _ in os.walk(TRAIN_RAWDATA_PATH):
        for label in label_list:
            print(label)

            # read each video in each label
            for dirname, _, filenames in os.walk(os.path.join(TRAIN_RAWDATA_PATH, label)):
                for filename in filenames:
                    if filename.endswith(VIDEO_EXTENSION):
                        video_path = os.path.join(dirname, filename)
                        print("\tpreprocessing: " +video_path)

                        ## already preprocess yet:  
                        feature_path = os.path.join(TRAIN_DATA_PATH, label, os.path.splitext(filename)[0]+"_sv_0.npy")
                        if os.path.isfile(feature_path):
                            continue

                        cap = cv2.VideoCapture(video_path)

                        no_frames =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        no_sub_video = no_frames // PROCESS_VIDEO_LENGTH
                        # process each sub video
                        for sub_video_counter in range(no_sub_video):
                            frame_list = []
                            path = os.path.join(TRAIN_DATA_PATH, label, os.path.splitext(filename)[0]+"_sv_"+str(sub_video_counter))
                            print("\t\tconvert into: " + path)
                            # process each image
                            for img_counter in range(PROCESS_VIDEO_LENGTH):
                                # read each frame
                                ret, frame = cap.read()  

                                # only get frame of each FRAME_STEP frames
                                if img_counter % FRAME_STEP != 0:
                                    continue
                                
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
                                frame_feature = np.append(feature_vector, [label_map[label]] )
                                frame_list.append(frame_feature)

                            if (len(frame_list) == 0):
                                print("This video part is corrupted, can not preprocess")
                                break
                            else:
                                np.save(path, frame_list)

def prepare_traindata_destination():
    if(not os.path.exists(TRAIN_RAWDATA_PATH)):
        os.makedirs(TRAIN_RAWDATA_PATH)
    if(not os.path.exists(TRAIN_DATA_PATH)):
        os.makedirs(TRAIN_DATA_PATH)
    for _, label_list, _ in os.walk(TRAIN_RAWDATA_PATH):
        for label in label_list:
            label_path = os.path.join(TRAIN_DATA_PATH, label)
            if(not os.path.exists(label_path)):
                os.makedirs(label_path)

def clear_train_data():
    for _, label_list, _ in os.walk(TRAIN_DATA_PATH):
        for label in label_list:
            for dirname, _, filenames in os.walk(os.path.join(TRAIN_DATA_PATH, label)):
                for filename in filenames:
                    if filename.endswith(DATA_EXTENSION):
                        video_path = os.path.join(dirname, filename)
                        print("Deleted: " +video_path)
                        os.remove(video_path)

def prepare_testdata_destination():
    if(not os.path.exists(TEST_RAWDATA_PATH)):
        os.makedirs(TEST_RAWDATA_PATH)
    if(not os.path.exists(TEST_DATA_PATH)):
        os.makedirs(TEST_DATA_PATH)

def extract_test_video():
    for dirname, _, filenames in os.walk(TEST_RAWDATA_PATH):
        for filename in filenames:
            if filename.endswith(VIDEO_EXTENSION):
                video_path = os.path.join(dirname, filename)
                print("\tpreprocessing: " +video_path)

                ## already preprocess:  
                path = os.path.join(TEST_DATA_PATH, os.path.splitext(filename)[0] + ".npy")
                if os.path.isfile(path):
                    continue

                # reading frame list
                cap = cv2.VideoCapture(video_path)


                frame_list = []
                # Check if camera opened successfully
                if (cap.isOpened()== False): 
                    print("Error opening video file")
                
                # Read until video is completed
                while(cap.isOpened()):
                    # Capture frame-by-frame
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # only get frame of each FRAME_STEP frames
                    current_frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    # print(current_frame_num)
                    if current_frame_num % FRAME_STEP != 0:
                        continue

                    # extract feature from frame
                    resize_frame = cv2.resize(frame, (120, 180), interpolation = cv2.INTER_AREA)
                    feature_vector, hog_img = hog(
                        resize_frame, pixels_per_cell=(14,14), 
                        cells_per_block=(2, 2), 
                        orientations=9, 
                        visualize=True, 
                        block_norm='L2-Hys',
                        feature_vector=True)

                    # adding label (y) in data
                    frame_list.append(feature_vector)


                if (len(frame_list) == 0):
                    print("This video part is corrupted, can not preprocess")
                else:
                    np.save(path, frame_list)

def run_preprocess_data_task(mode):
    if (mode ==  "train"):
        prepare_traindata_destination()                
        clear_train_data()
        extract_train_video()
    elif (mode == "test"):
        prepare_testdata_destination()
        extract_test_video()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default="train")               # train/test
    args = parser.parse_args()

    run_preprocess_data_task(args.mode)
