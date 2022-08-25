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

def divide_dataset_from_file(input_file, train_file, test_file, val_file, valid_ratio=VALID_RATIO, test_ratio=TEST_RATIO):
    #Divide dataset (from the input file) into train set, test set, and validation set
    image_list = [line for line in open(input_file)]
    total_seqs = len(image_list)
    permutation = np.random.RandomState(27731).permutation(total_seqs)
    valid_size = int(valid_ratio*total_seqs)
    test_size = int(test_ratio*total_seqs)

    ftrain=open(train_file,'w')
    ftest=open(test_file,'w')
    fval=open(val_file,'w')

    for i in range(total_seqs):
        item=image_list[permutation[i]]
        if i<valid_size:
            fval.write(item)
        elif i<valid_size+test_size:
            ftest.write(item)
        else:
            ftrain.write(item)

    ftrain.close()
    ftest.close()
    fval.close()

def seperate_dataset(rawdata_path, val_ratio=0.2, test_ratio=0.1,\
    file_list=None,train_file=None,test_file=None, val_file=None):
    if file_list is None:
        file_list=os.path.join(os.path.dirname(rawdata_path),'data_file.lst')
    if train_file is None:
        train_file=os.path.join(os.path.dirname(rawdata_path),'train_file.lst')
    if test_file is None:
        test_file=os.path.join(os.path.dirname(rawdata_path),'test_file.lst')
    if val_file is None:
        val_file=os.path.join(os.path.dirname(rawdata_path),'val_file.lst')
    if not os.path.exists(file_list):
        generate_list_file_layer1(output_file=file_list,rawdata_path=rawdata_path)
    if not os.path.exists(test_file) and not os.path.exists(train_file):
        divide_dataset_from_file(input_file=file_list,train_file=train_file,test_file=test_file,\
            val_file=val_file, valid_ratio=val_ratio,test_ratio=test_ratio)
    
    return


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

def generate_val_from_train_test(org_file, train_file, test_file, val_file):
    image_list = [line for line in open(org_file)]
    image_list_train = [line for line in open(train_file)]
    image_list_test = [line for line in open(test_file)]
    image_list_val=[]
    for item in image_list:
        print('Processing:',item)
        if item in image_list_train:
            istrain=True
        else:
            istrain=False
        if item in image_list_test:
            istest=True
        else:
            istest=False
        if istrain or istest:
            continue
        image_list_val.append(item)
    if len(image_list)!=0:
        f=open(val_file,'w')
        for line in image_list_val:
            f.write(line)
        f.close()


if __name__ == '__main__':

    '''
    seperate_dataset(rawdata_path=LAYER_1_RAWDATA_PATH,val_ratio=0.2, test_ratio=0.1)
    '''

    '''
    input_file='data/layer_1_data/data_file.lst'
    train_file_l1='data/layer_1_data/tmp_train.lst'
    test_file_l1='data/layer_1_data/tmp_test.lst'
    val_file_l1='data/layer_1_data/tmp_val.lst'
    divide_dataset_from_file(input_file=input_file,train_file=train_file_l1,\
        test_file=test_file_l1,val_file=val_file_l1)
    '''
    #Phục hồi lại file validation
    '''
    org_file_l1='data/layer_2_data/data_file.lst'
    train_file_l1='data/layer_2_data/train.lst'
    test_file_l1='data/layer_2_data/test4.lst'
    val_file_l1='data/layer_2_data/val4.lst'
    generate_val_from_train_test(org_file=org_file_l1,train_file=train_file_l1,\
        test_file=test_file_l1,val_file=val_file_l1)
    '''
    
    #Tạo danh sách các file từ tập dữ liệu 1
    '''
    output_file='data/layer_1_data/data_file.lst'
    generate_list_file_layer1(output_file=output_file, rawdata_path=LAYER_1_RAWDATA_PATH)
    '''
    
    #Tạo danh sách các file từ tập dữ liệu 2
    '''
    output_file='data/layer_2_data/data_file.lst'
    generate_list_file_layer2(output_file=output_file, rawdata_path=LAYER_2_RAWDATA_PATH)
    '''

    #Phục hồi file đặc trưng về file danh mục cho lớp 1
    '''
    rawdata_path_l1='data/layer_1_data/raw_video'

    path_l1='data/layer_1_data/samples/case2/test'
    output_file_l1='data/layer_1_data/test.lst'
    output_file_l1_without_aug='data/layer_1_data/test_f.lst'

    get_file_list_layer1(path=path_l1,output_file=output_file_l1,rawdata_path=rawdata_path_l1,output_file2=output_file_l1_without_aug)
    '''
    #Phục hồi file đặc trưng về file danh mục cho lớp 2
    '''
    path_l2='data/test/case2/test4'
    output_file_l2='data/layer_2_data/test4.lst'
    rawdata_path_l2='data/layer_2_data/raw_video'
    get_file_list_layer2(path=path_l2,output_file=output_file_l2,rawdata_path=rawdata_path_l2)
    '''
    
