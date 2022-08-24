import os
import cv2
import numpy as np

from property import *

layer1_folder_map = {
    "A_01":"Step_1",
    "A_02":"Step_2_Left",
    "A_03":"Step_2_Right",
    "A_04":"Step_3",
    "A_05":"Step_4_Left",
    "A_06":"Step_4_Right",
    "A_07":"Step_5_Left",
    "A_08":"Step_5_Right",
    "A_09":"Step_6_Left",
    "A_10":"Step_6_Right",
}

def generate_list_file_layer1(output_file, rawdata_path):
    #Generate the list of files with augmentation information
    #For Layer 1
    #read each label video
    f=open(output_file,'w')
    for _, label_list, _ in os.walk(rawdata_path):
        for label in label_list:
            # read each video in each label
            for dirname, _, filenames in os.walk(os.path.join(rawdata_path, label)):
                for filename in filenames:
                    if not filename.endswith(VIDEO_EXTENSION):
                        continue

                    video_path = os.path.join(dirname, filename)
                    print("\tProcessing: " + video_path)
                    cap = cv2.VideoCapture(video_path)
                    no_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    no_sub_video = no_frames // PROCESS_VIDEO_LENGTH
                    for rotate in ["0","90","180","270"]:
                        for sub_video in range(no_sub_video):
                            for frame_step in range(FRAME_STEP):
                                f.write(video_path)
                                f.write("\tsub_video:"+str(sub_video)+"\trotate:"+rotate+"\tframe_step:"+str(frame_step)+"\n")
    f.close()

def generate_list_file_layer2(output_file, rawdata_path):
    #Generate the list of files with augmentation information
   # read each label video
    f=open(output_file,'w')
    for _, label_list, _ in os.walk(rawdata_path):
        for label in label_list:
            # read each video in each label
            for dirname, _, filenames in os.walk(os.path.join(rawdata_path, label)):
                for filename in filenames:
                    if not filename.endswith(VIDEO_EXTENSION):
                        continue
                    video_path = os.path.join(dirname, filename)
                    print("\tProcessing: " + video_path)
                    f.write(video_path+'\t'+label+'\n')
    f.close()

def divide_dataset_from_file(input_file, train_file, test_file, valid_file):
    #Divide dataset into train set, test set, and validation set
    image_list = [line.strip().split() for line in open(input_file)]
    total_seqs = len(image_list)
    permutation = np.random.RandomState(27731).permutation(total_seqs)
    valid_size = int(VALID_RATIO*total_seqs)
    test_size = int(TEST_RATIO*total_seqs)

    valid_list=image_list[:,permutation[:valid_size]]
    test_list = image_list[:,permutation[valid_size:valid_size+test_size]]
    train_list = image_list[:,permutation[valid_size+test_size:]]


def get_file_list_layer1(path,output_file,rawdata_path,output_file2):
    #path is a folder contains list of files
    list_file=[]
    not_exist_files=[]
    #Two output files: 1. with augmentation, 2. wo_aug_file for wihout augmentation
    f=open(output_file,'w') #augmentation
    f2=open(output_file2,'w') #wihout augmentation
    for _, _, filenames in os.walk(path):
        for filename in filenames:
            if not filename.endswith(DATA_EXTENSION):
                continue
            file_names=filename.split('.')[0].split('_')
            l=len(file_names)
            file_name='_'.join([str(file_names[i]) for i in range(l-5)])
            label_in_filename='_'.join([str(file_names[i]) for i in range(2,4)])
            label=layer1_folder_map[label_in_filename]
            #Augmentation information
            sub_video=file_names[-4]
            rotate=file_names[-2]
            frame_step=file_names[-1]
            is_exist=False
            for ext in VIDEO_EXTENSION:
                file_name_ext=file_name+ext
                rawfile=os.path.join(rawdata_path, label, file_name_ext)
                if os.path.exists(rawfile):
                    is_exist=True
                    f.write(rawfile)
                    f.write("\tsub_video:"+sub_video+"\trotate:"+rotate+"\tframe_step:"+frame_step+"\n")
                    if file_name_ext not in list_file:
                        list_file.append(file_name_ext)
                        f2.write(rawfile+"\n")
            if not is_exist:
                not_exist_files.append(filename)
    f.close()
    f2.close()

def get_file_list_layer2(path,output_file,rawdata_path):
    f=open(output_file,'w')
    for _, _, filenames in os.walk(path):
        for filename in filenames:
            if not filename.endswith(DATA_EXTENSION):
                continue
            label,file_name=filename.split('.')[0].split('_',1)
            for ext in VIDEO_EXTENSION:
                file_name_ext=file_name+ext
                rawfile=os.path.join(rawdata_path, label, file_name_ext)
                if os.path.exists(rawfile):
                    f.write(rawfile+'\t'+label+'\n')
    f.close()

if __name__ == '__main__':

    input_file='data/layer_1_data/data_file.lst'
    divide_dataset_from_file(input_file=input_file)
    '''
    output_file='data/layer_1_data/data_file.lst'
    generate_list_file_layer1(output_file=output_file, rawdata_path=LAYER_1_RAWDATA_PATH)
    '''

    '''
    output_file='data/layer_2_data/data_file.lst'
    generate_list_file_layer2(output_file=output_file, rawdata_path=LAYER_2_RAWDATA_PATH)
    '''
    '''
    rawdata_path_l1='data/layer_1_data/raw_video'

    path_l1='data/layer_1_data/samples/case1/test'
    output_file_l1='data/layer_1_data/case1_test.lst'
    output_file_l1_without_aug='data/layer_1_data/case1_test_org.lst'

    get_file_list_layer1(path=path_l1,output_file=output_file_l1,rawdata_path=rawdata_path_l1,output_file2=output_file_l1_without_aug)
    '''
    '''
    path_l2='data/test/case1/train'
    output_file_l2='data/layer_2_data/case1_train.lst'
    rawdata_path_l2='data/layer_2_data/raw_video'
    get_file_list_layer2(path=path_l2,output_file=output_file_l2,rawdata_path=rawdata_path_l2)
    '''
    
