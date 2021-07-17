import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

## import config property
from property import *

class VideoSet:
    def __init__(self):
        for dirname, _, filenames in os.walk(os.path.join(TEST_DATA_PATH)):
            # read each video file in each label
            for filename in filenames:
                if filename.endswith(DATA_EXTENSION):
                    data_path = os.path.join(dirname, filename)
                    # print("\treading: " +data_path)
                    video_feature = np.load(data_path)
                    print(video_feature.shape)


if __name__ == '__main__':
    video_set = VideoSet()
    
