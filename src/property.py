import os

### PROPERTY ###
######
TRAIN_RAWDATA_PATH = os.path.join("..", "data", "train", "raw_video")
TRAIN_DATA_PATH = os.path.join("..", "data", "train", "clean_data")
TEST_RAWDATA_PATH = os.path.join("..", "data", "test", "raw_video")
TEST_DATA_PATH = os.path.join("..", "data", "test", "clean_data")

VIDEO_EXTENSION = ('.mp4', '.avi')
DATA_EXTENSION = ('.npy')
######

### Preprocessing hyperparameter ###
######
PROCESS_VIDEO_LENGTH = 30   # frames 
FRAME_STEP = 3              # only take 1 every 3 frames
WINDOWS_LENGTH = PROCESS_VIDEO_LENGTH / FRAME_STEP  # get 1 second and only get 10 frame per second
######

### Training hyperparameter ###
######
VALID_RATIO = 0.06
TEST_RATIO = 0.03
BATCH_SIZE = 2

NUMBER_OF_TREE = 12
######

### Model parameter ###
######
MODEL_TYPE = "lstm"       # type of Cell for network
MODEL_SIZE = 10           # time step
MODEL_EPOCH_NUM = 200     # iterative
MODEL_LOG_PERIOD = 1       # number of iterative for each save
MODEL_SPARSITY = 0.0
######

