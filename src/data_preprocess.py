from property import *
from file_lib import *
from maps import *
from read_data import _get_next_frame
import cv2
import sys
import os

def prepare_data_path(rawdata_path, data_path):
    if (not os.path.exists(rawdata_path)):
        return
    for _, label_list, _ in os.walk(rawdata_path):
        for label in label_list:
            label_path = os.path.join(data_path, label)
            if(not os.path.exists(label_path)):
                os.makedirs(label_path)

def preprocess_layer1_video():
    prepare_data_path(rawdata_path=LAYER_1_RAWDATA_PATH,
                      data_path=LAYER_1_TRAIN_PATH)
    prepare_data_path(rawdata_path=LAYER_1_RAWDATA_PATH,
                      data_path=LAYER_1_VALID_PATH)                  
    prepare_data_path(rawdata_path=LAYER_1_RAWDATA_PATH,
                      data_path=LAYER_1_TEST_PATH)
    
    layer1_video_to_feature(input_file=LAYER_1_TRAIN_FILE,
                            output_folder=LAYER_1_TRAIN_PATH)
    layer1_video_to_feature(input_file=LAYER_1_VALID_FILE,
                            output_folder=LAYER_1_VALID_PATH)
    layer1_video_to_feature(input_file=LAYER_1_TEST_FILE,
                            output_folder=LAYER_1_TEST_PATH)

def preprocess_layer2_video():
    prepare_data_path(rawdata_path=LAYER_2_RAWDATA_PATH,
                      data_path=LAYER_2_TRAIN_PATH)
    prepare_data_path(rawdata_path=LAYER_2_RAWDATA_PATH,
                      data_path=LAYER_2_VALID_PATH)
    prepare_data_path(rawdata_path=LAYER_2_RAWDATA_PATH,
                      data_path=LAYER_2_TEST_PATH)

    prepare_data_path(rawdata_path=LAYER_2_TRAIN_PATH,
                      data_path=LAYER_2_TRAIN_PROCESSED_PATH)
    prepare_data_path(rawdata_path=LAYER_2_VALID_PATH,
                      data_path=LAYER_2_VALID_PROCESSED_PATH)
    prepare_data_path(rawdata_path=LAYER_2_TEST_PATH,
                      data_path=LAYER_2_TEST_PROCESSED_PATH)

    layer2_video_to_feature(input_file=LAYER_2_TRAIN_FILE,
                            output_folder=LAYER_2_TRAIN_PATH)
    layer2_video_to_feature(input_file=LAYER_2_VALID_FILE,
                            output_folder=LAYER_2_VALID_PATH)
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
    return

def data_preprocess(divide_layer1_data=False, divide_layer2_data=False,\
    generate_layer1_feature_data=REGENERATE_LAYER1_DATA, generate_layer2_feature_data=REGENERATE_LAYER2_DATA):
    if divide_layer1_data:
        seperate_dataset(rawdata_path=LAYER_1_RAWDATA_PATH,val_ratio=0.2, test_ratio=0.1,\
            train_file=LAYER_1_TRAIN_FILE, test_file=LAYER_1_TEST_FILE, val_file=LAYER_1_VALID_FILE)

    if generate_layer1_feature_data:
        preprocess_layer1_video()

    if divide_layer2_data:
        seperate_dataset(rawdata_path=LAYER_2_RAWDATA_PATH,val_ratio=0.2, test_ratio=0.1,\
            train_file=LAYER_2_TRAIN_FILE, test_file=LAYER_2_TEST_FILE, val_file=LAYER_2_VALID_FILE)

    if generate_layer2_feature_data:
        preprocess_layer2_video()
    
    return

if __name__ == '__main__':

 
    data_preprocess(divide_layer1_data=True,generate_layer1_feature_data=True)