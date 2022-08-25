import tensorflow as tf
from property import *
from skimage.feature import hog
import cv2
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from matplotlib import pyplot as plt
import sys
import os
from warnings import simplefilter

from skimage import feature
simplefilter(action='ignore', category=FutureWarning)


def extract_single_vid(video_path):
    # reading frame list
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0

    frame_list = []
    # Check if camera opened successfully
    if (cap.isOpened() == False): 
        print("Error opening video file")

    # Read until video is completed
    while(cap.isOpened()):
        is_continue, feature_vector = _get_next_frame(cap, frame_counter)
        frame_counter += 1

        if feature_vector is not None:
            pass
        elif is_continue:
            continue
        else:
            break

        frame_list.append(feature_vector)

    return frame_list


def _get_next_frame(cap, counter, rotation_direction=None):
    # read each frame
    ret, frame = cap.read()

    if not ret:
        frame_list = []   # not using this sub_video 
        return (False, None)

    # preprocess each frame
    resize_frame = cv2.resize(frame, (180, 120), interpolation= cv2.INTER_AREA)
    if rotation_direction is not None:
        resize_frame = cv2.rotate(resize_frame, rotation_direction)

    # extract feature from image
    # TODO Adapt the parameter and preprocessing/scale the image
    feature_vector, hog_img = hog(
        resize_frame, pixels_per_cell=(15, 15), 
        cells_per_block=(2, 2),
        orientations=9,
        visualize=True,
        block_norm='L2-Hys',
        feature_vector=True)

    return (True, feature_vector)

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


def clear_data(data_path):
    for _, label_list, _ in os.walk(data_path):
        for label in label_list:
            for dirname, _, filenames in os.walk(os.path.join(data_path, label)):
                for filename in filenames:
                    if filename.endswith(DATA_EXTENSION):
                        video_path = os.path.join(dirname, filename)
                        print("Deleted: " + video_path)
                        os.remove(video_path)

