import os

### PROPERTY ###
######
LAYER_1_TRAIN_RAWDATA_PATH = os.path.join("..", "data", "layer_1_train", "raw_video")
LAYER_1_TRAIN_DATA_PATH = os.path.join("..", "data", "layer_1_train", "clean_data")
LAYER_2_TRAIN_RAWDATA_PATH = os.path.join("..", "data", "layer_2_train", "raw_video")
LAYER_2_TRAIN_DATA_PATH = os.path.join("..", "data", "layer_2_train", "clean_data")
LAYER_2_PROCESSED_DATA_PATH = os.path.join("..", "data", "layer_2_train", "processed_data")
TEST_RAWDATA_PATH = os.path.join("..", "data", "test", "raw_video")
TEST_DATA_PATH = os.path.join("..", "data", "test", "clean_data")

VIDEO_EXTENSION = ('.mp4', '.avi')
DATA_EXTENSION = ('.npy')

REGENERATE_LAYER1_DATA = False                                      # Re-compute dataset flag
REGENERATE_LAYER2_DATA = False                                      # Re-compute dataset flag
######

### Preprocessing hyperparameter ###
######
PROCESS_VIDEO_LENGTH = 30                                           # frames per sub video to train
FRAME_STEP = 3                                                      # only take 1 every 3 frames
WINDOWS_LENGTH = int(PROCESS_VIDEO_LENGTH / FRAME_STEP)             # length of sliding windows when reading full video
######

### Layer 1 Model hyperparameter ###
######
VALID_RATIO = 0.06                                                  # Valid percentage when dividing dataset
TEST_RATIO = 0.03                                                   # Test percentage when dividing dataset
BATCH_SIZE = 64                                                     # Batch size fitting in network
NUMBER_OF_TREE = 12                                                 # Number of tree in random forest (default: 12)

MODEL_TYPE = "ltc"                                                  # type of Cell for network (lstm / ltc / ltc_ex / ltc_rk / node / ctgru / ctrnn)
MODEL_SIZE = 10                                                     # time step
MODEL_EPOCH_NUM = 50                                                # iterative
MODEL_LOG_PERIOD = 1                                                # number of iterative for each save
MODEL_SPARSITY = 0.0
######

### Layer 2 Model parameter ###
######
LAYER_2_MODEL_TYPE = "mlp"                                          # Type of layer 2 model
LAYER_2_VALID_RATIO = 0.2                                           # Valid percentage when dividing dataset
LAYER_2_TEST_RATIO = 0.2                                            # Test percentage when dividing dataset

LAYER2_EPOCH_NUM = 500                                              # Number of epoch when training layer2
######