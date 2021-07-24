from warnings import simplefilter

from skimage import feature 
simplefilter(action='ignore', category=FutureWarning)

import os
import glob
import argparse
import sys

from matplotlib import pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import cv2
from skimage.feature import hog

## import config property
from property import *

tf.logging.set_verbosity(tf.logging.ERROR)

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


video_map = {"no": 0, "yes": 1}


def extract_layer1_train_video():
    # read each label video
    for _, label_list, _ in os.walk(LAYER_1_TRAIN_RAWDATA_PATH):
        for label in label_list:
            print(label)

            # read each video in each label
            for dirname, _, filenames in os.walk(os.path.join(LAYER_1_TRAIN_RAWDATA_PATH, label)):
                for filename in filenames:
                    if filename.endswith(VIDEO_EXTENSION):
                        video_path = os.path.join(dirname, filename)
                        print("\tpreprocessing: " +video_path)

                        ## already preprocess yet:  
                        feature_path = os.path.join(LAYER_1_TRAIN_DATA_PATH, label, os.path.splitext(filename)[0]+"_sv_0.npy")
                        if os.path.isfile(feature_path):
                            continue

                        cap = cv2.VideoCapture(video_path)

                        no_frames =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        no_sub_video = no_frames // PROCESS_VIDEO_LENGTH
                        # process each sub video
                        for sub_video_counter in range(no_sub_video):
                            frame_list = []
                            path = os.path.join(LAYER_1_TRAIN_DATA_PATH, label, os.path.splitext(filename)[0]+"_sv_"+str(sub_video_counter)+".npy")
                            print("\t\tconvert into: " + path)
                            # process each image
                            for img_counter in range(PROCESS_VIDEO_LENGTH):
                                # read each frame
                                is_continue, feature_vector = _get_next_frame(cap, img_counter)

                                if is_continue == None:
                                    pass
                                elif is_continue:
                                    continue
                                else:
                                    break
                                # adding label (y) in data
                                frame_feature = np.append(feature_vector, [label_map[label]] )
                                frame_list.append(frame_feature)

                            if (len(frame_list) == 0):
                                print("This video part is corrupted, can not preprocess")
                                break
                            else:
                                np.save(path, frame_list)

def extract_layer2_train_video():
    # read each video
    for _, video_label_list, _ in os.walk(LAYER_2_TRAIN_RAWDATA_PATH):
        for video_label in video_label_list:
            for dirname, _, filenames in os.walk(os.path.join(LAYER_2_TRAIN_RAWDATA_PATH, video_label)):
                for filename in filenames:
                    if filename.endswith(VIDEO_EXTENSION):
                        video_path = os.path.join(dirname, filename)
                        print("\tpreprocessing: " +video_path)
                        sys.stdout.flush()

                        ## already preprocess:  
                        path = os.path.join(LAYER_2_TRAIN_DATA_PATH, video_label, os.path.splitext(filename)[0] + ".npy")
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
                            is_continue, feature_vector = _get_next_frame(cap)

                            if feature_vector != None:
                                pass
                            elif is_continue:
                                continue
                            else:
                                break

                            frame_feature = np.append(feature_vector, video_map[video_label] )
                            # adding label (y) in data
                            frame_list.append(frame_feature)

                        if (len(frame_list) == 0):
                            print("This video part is corrupted, can not preprocess")
                        else:
                            np.save(path, frame_list)

def extract_single_vid(video_path):
    # reading frame list
    cap = cv2.VideoCapture(video_path)
    frame_list = []
    # Check if camera opened successfully
    if (cap.isOpened()== False): 
        print("Error opening video file")
    
    # Read until video is completed
    while(cap.isOpened()):
        is_continue, feature_vector = _get_next_frame(cap)

        if feature_vector != None:
            pass
        elif is_continue:
            continue
        else:
            break
        
        frame_list.append(feature_vector)
    
    return frame_list


def _get_next_frame(cap, counter=None):
    if counter == None:
        counter = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # only get frame of each FRAME_STEP frames
    if counter % FRAME_STEP != 0:
        return (True, None)
    
    # read each frame
    ret, frame = cap.read()  

    if not ret:
        frame_list=[]   # not using this sub_video 
        return (False, None)
        
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

    return (None, feature_vector)

def prepare_traindata_destination(rawdata_path, data_path):
    if (not os.path.exists(rawdata_path)):
        os.makedirs(rawdata_path)
    if (not os.path.exists(data_path)):
        os.makedirs(data_path)
    for _, label_list, _ in os.walk(rawdata_path):
        for label in label_list:
            label_path = os.path.join(data_path, label)
            if(not os.path.exists(label_path)):
                os.makedirs(label_path)

def prepare_testdata_destination():
    if (not os.path.exists(TEST_RAWDATA_PATH)):
        os.makedirs(TEST_RAWDATA_PATH)
    if (not os.path.exists(TEST_DATA_PATH)):
        os.makedirs(TEST_DATA_PATH)

def prepare_processed_data_destination():
    if ( not os.path.exists(LAYER_2_PROCESSED_DATA_PATH)): 
        os.makedirs(LAYER_2_PROCESSED_DATA_PATH)

    for dirname, _, filenames in os.walk(LAYER_2_PROCESSED_DATA_PATH):
        for filename in filenames:
            if filename.endswith(DATA_EXTENSION):
                video_path = os.path.join(dirname, filename)
                print("Deleted: " +video_path)
                os.remove(video_path)            

def clear_train_data(data_path):
    for _, label_list, _ in os.walk(data_path):
        for label in label_list:
            for dirname, _, filenames in os.walk(os.path.join(data_path, label)):
                for filename in filenames:
                    if filename.endswith(DATA_EXTENSION):
                        video_path = os.path.join(dirname, filename)
                        print("Deleted: " +video_path)
                        os.remove(video_path)

def write_log(string):
    print(string)
    sys.stdout.flush()


