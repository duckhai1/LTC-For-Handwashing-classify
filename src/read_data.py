import tensorflow.compat.v1 as tf  # Hieu 22/8
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


#import tensorflow as tf
tf.disable_v2_behavior()  # Hieu 22/8


# import config property

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

rotation_map = {
    "rotate_0": None,
    "rotate_90": cv2.ROTATE_90_COUNTERCLOCKWISE,
    "rotate_180": cv2.ROTATE_180,
    "rotate_270": cv2.ROTATE_90_CLOCKWISE
}


def preprocess_layer1_video():
    prepare_data_path(rawdata_path=LAYER_1_RAWDATA_PATH,
                      data_path=LAYER_1_TRAIN_PATH)
    prepare_data_path(rawdata_path=LAYER_1_RAWDATA_PATH,
                      data_path=LAYER_1_TEST_PATH)

    layer1_video_to_feature(input_file=LAYER_1_TRAIN_FILE,
                            output_folder=LAYER_1_TRAIN_PATH)
    layer1_video_to_feature(input_file=LAYER_1_TEST_FILE,
                            output_folder=LAYER_1_TEST_PATH)

def preprocess_layer2_video():
    prepare_data_path(rawdata_path=LAYER_2_RAWDATA_PATH,
                      data_path=LAYER_2_TRAIN_PATH)
   prepare_data_path(rawdata_path=LAYER_2_RAWDATA_PATH,
                      data_path=LAYER_2_TRAIN_PATH)

    layer2_video_to_feature(input_file=LAYER_2_TRAIN_FILE,
                            output_folder=LAYER_2_TRAIN_PATH)
    layer2_video_to_feature(input_file=LAYER_2_TEST_FILE,
                            output_folder=LAYER_2_TEST_PATH)

def layer1_video_to_feature(input_file, output_folder):
    image_list = [line.strip().split() for line in open(input_file)]
    for item in image_list:
        video_path = item[0]
        sub_video = item[1].split(':')[1]
        rotate_direction = item[2].replace(':', '_')
        frame_step = item[3].split(':')[1]
        print("\tpreprocessing: " + video_path)
        filename = video_path.split('/')[-1]
        label = video_path.split('/')[-2]
        feature_path = os.path.join(output_folder, os.path.splitext(
            filename)[0]+"_sv_"+sub_video+"_"+rotate_direction + "_"+frame_step + ".npy")
        if os.path.isfile(feature_path):
            continue
        cap = cv2.VideoCapture(video_path)

        #no_frames =  int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        #no_sub_video = no_frames // PROCESS_VIDEO_LENGTH
        for i in range(int(sub_video)*PROCESS_VIDEO_LENGTH):  # Skip previous subvideos
            _, _ = cap.read()

        frame_dic_step = []

        # process each image
        is_break = False
        for img_counter in range(PROCESS_VIDEO_LENGTH):
            if img_counter % FRAME_STEP != int(frame_step):
                _, _ = cap.read()
                continue

            # read each frame
            is_continue, feature_vector = _get_next_frame(
                cap, counter=img_counter, rotation_direction=rotation_map[rotate_direction])

            if is_continue == True:
                pass
            elif is_continue == False:
                is_break = True
                break
            # adding label (y) in data
            frame_feature = np.append(feature_vector, [label_map[label]])
            frame_dic_step.append(frame_feature)
            if is_break:
                break

        if is_break:
            print("This video part is corrupted, can not preprocess")
            break
        else:
            path = os.path.join(output_folder, label, os.path.splitext(filename)[
                                0]+"_sv_"+sub_video+"_"+rotate_direction+"_"+frame_step+".npy")
            print("\t\tconvert into: " + path)
            np.save(path, frame_dic_step)


def layer2_video_to_feature(input_file, output_folder):
    image_list = [line.strip().split() for line in open(input_file)]
    for item in image_list:
        video_path = item[0]
        print("\tpreprocessing: " + video_path)
        sys.stdout.flush()
        filename = video_path.split('/')[-1]
        label = item[1]
        # already preprocess:
        path = os.path.join(output_folder, label,
                            os.path.splitext(filename)[0] + ".npy")
        if os.path.isfile(path):
            continue
        # reading frame list
        cap = cv2.VideoCapture(video_path)
        frame_counter = 0

        frame_list = []
        # Check if camera opened successfully
        if (cap.isOpened() == False):
            print("Error opening video file")
        # Read until video is completed
        while(cap.isOpened()):
            # Capture frame-by-frame
            is_continue, feature_vector = _get_next_frame(
                cap, counter=frame_counter)
            frame_counter += 1

            if feature_vector is not None:
                pass
            elif is_continue:
                continue
            else:
                break

            frame_feature = np.append(
                feature_vector, video_map[label])
            # adding label (y) in data
            frame_list.append(frame_feature)

        if (len(frame_list) == 0):
            print("This video part is corrupted, can not preprocess")
        else:
            np.save(path, frame_list)


def extract_layer1_train_video():
    # read each label video
    for _, label_list, _ in os.walk(LAYER_1_TRAIN_RAWDATA_PATH):
        for label in label_list:
            print(label)

            # read each video in each label
            for dirname, _, filenames in os.walk(os.path.join(LAYER_1_TRAIN_RAWDATA_PATH, label)):
                for filename in filenames:
                    if not filename.endswith(VIDEO_EXTENSION):
                        continue

                    video_path = os.path.join(dirname, filename)
                    print("\tpreprocessing: " + video_path)
                    for rotate_direction in rotation_map.keys():
                        # already preprocess yet:
                        feature_path = os.path.join(LAYER_1_TRAIN_DATA_PATH, label, os.path.splitext(filename)[0]+"_sv_0_"+rotate_direction + "_0" + ".npy")
                        if os.path.isfile(feature_path):
                            continue

                        cap = cv2.VideoCapture(video_path)

                        no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        no_sub_video = no_frames // PROCESS_VIDEO_LENGTH
                        # process each sub video
                        for sub_video_counter in range(no_sub_video):
                            frame_dic = {}
                            for i in range(FRAME_STEP):
                                frame_dic[i] = []

                            # process each image
                            is_break = False
                            for img_counter in range(PROCESS_VIDEO_LENGTH):
                                for i in range(FRAME_STEP):
                                    if img_counter % FRAME_STEP != i:
                                        continue

                                    # read each frame
                                    is_continue, feature_vector = _get_next_frame(
                                        cap, counter=img_counter, rotation_direction=rotation_map[rotate_direction])

                                    if is_continue == True:
                                        pass
                                    elif is_continue == False:
                                        is_break = True
                                        break
                                    # adding label (y) in data
                                    frame_feature = np.append(feature_vector, [label_map[label]])
                                    frame_dic[i].append(frame_feature)
                                if is_break:
                                    break

                            if is_break:
                                print(
                                    "This video part is corrupted, can not preprocess")
                                break
                            else:
                                for i in range(FRAME_STEP):
                                    path = os.path.join(LAYER_1_TRAIN_DATA_PATH, label, os.path.splitext(filename)[
                                                        0]+"_sv_"+str(sub_video_counter)+"_"+rotate_direction+"_"+str(i)+".npy")
                                    print("\t\tconvert into: " + path)
                                    np.save(path, frame_dic[i])


def extract_layer2_train_video():
    # read each video
    for _, video_label_list, _ in os.walk(LAYER_2_TRAIN_RAWDATA_PATH):
        for video_label in video_label_list:
            for dirname, _, filenames in os.walk(os.path.join(LAYER_2_TRAIN_RAWDATA_PATH, video_label)):
                for filename in filenames:
                    if filename.endswith(VIDEO_EXTENSION):
                        video_path = os.path.join(dirname, filename)
                        print("\tpreprocessing: " + video_path)
                        sys.stdout.flush()

                        # already preprocess:
                        path = os.path.join(
                            LAYER_2_TRAIN_DATA_PATH, video_label, os.path.splitext(filename)[0] + ".npy")
                        if os.path.isfile(path):
                            continue

                        # reading frame list
                        cap = cv2.VideoCapture(video_path)
                        frame_counter = 0

                        frame_list = []
                        # Check if camera opened successfully
                        if (cap.isOpened() == False): 
                            print("Error opening video file")

                        # Read until video is completed
                        while(cap.isOpened()):
                            # Capture frame-by-frame
                            is_continue, feature_vector = _get_next_frame(
                                cap, counter=frame_counter)
                            frame_counter += 1

                            if feature_vector is not None:
                                pass
                            elif is_continue:
                                continue
                            else:
                                break

                            frame_feature = np.append(feature_vector, video_map[video_label])
                            # adding label (y) in data
                            frame_list.append(frame_feature)

                        if (len(frame_list) == 0):
                            print("This video part is corrupted, can not preprocess")
                        else:
                            np.save(path, frame_list)


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

def prepare_data_path(rawdata_path, data_path):
    if (not os.path.exists(rawdata_path)):
        return
    for _, label_list, _ in os.walk(rawdata_path):
        for label in label_list:
            label_path = os.path.join(data_path, label)
            if(not os.path.exists(label_path)):
                os.makedirs(label_path)


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
    if (not os.path.exists(LAYER_2_PROCESSED_DATA_PATH)): 
        os.makedirs(LAYER_2_PROCESSED_DATA_PATH)


def clear_processed_data_destination():
    for dirname, _, filenames in os.walk(LAYER_2_PROCESSED_DATA_PATH):
        for filename in filenames:
            if filename.endswith(DATA_EXTENSION):
                video_path = os.path.join(dirname, filename)
                print("Deleted: " + video_path)
                os.remove(video_path)


def clear_train_data(data_path):
    for _, label_list, _ in os.walk(data_path):
        for label in label_list:
            for dirname, _, filenames in os.walk(os.path.join(data_path, label)):
                for filename in filenames:
                    if filename.endswith(DATA_EXTENSION):
                        video_path = os.path.join(dirname, filename)
                        print("Deleted: " + video_path)
                        os.remove(video_path)


if __name__ == '__main__':
    preprocess_layer2_video()
#    preprocess_layer1_video()
