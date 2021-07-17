import os

import numpy as np
import pandas as pd
import tensorflow as tf

import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU

def debug_train():
    sess=tf.Session()    
    sess.run(tf.global_variables_initializer())
    B = tf.constant([[[2, 20, 30, 3, 6], 
                     [3, 11, 16, 1, 8],
                     [14, 45, 23, 5, 27]],
                    [[3, 10, 31, 67, 4], 
                     [13, 12, 6, 31, 48],
                     [1, 5, 21, 35, 25]]]
                 )

    A = tf.constant([[[-2.7271686,-1.4388094,-2.0188243,-1.0561708,1.637208,0.19901484,-0.8719948]]
,[[-3.7455251,-1.538672,-2.7955987,-1.0793971,2.3620052,-0.03944147,-1.448089]]
,[[-4.114932,-1.5860522,-3.0559874,-0.822364,2.6021085,-0.07498071,-1.8083875]]
,[[-4.1731024,-1.6080122,-3.100067,-0.7532509,2.6319382,-0.06349739,-1.8964169]]
,[[-4.174638,-1.6687288,-3.096071,-0.6641146,2.5824366,0.02426776,-1.9970801]]
,[[-4.1612864,-1.7045102,-3.0908556,-0.62772894,2.5533545,0.07228264,-2.0382054]]
,[[-4.0935173,-1.6301603,-3.1162238,-0.7790887,2.6157367,-0.06814519,-1.8748906]]
,[[-3.7928922,-1.9276924,-3.0089927,-0.64036053,2.4143994,0.27815792,-2.0768926]]
,[[-3.7215254,-2.0358167,-2.9838016,-0.5558922,2.3290517,0.41590202,-2.1761055]]
,[[-3.9860337,-1.6926246,-3.0895317,-0.7418579,2.57034,0.00664258,-1.9414456]]])
    print(sess.run(A).shape)
    
    # max = sess.run(tf.math.argmax(A, axis=2))
    # print(max)
    # print(sess.run(tf.argsort(A, axis=2)))
    beforeA = sess.run(tf.argsort(A, axis=2))
    print("beforeA", beforeA)
    afterA = beforeA.reshape([10,7])
    print("afterA", afterA)


def debug_read_data():
    TRAIN_RAWDATA_PATH = os.path.join("..", "data", "train", "raw_video")
    TRAIN_DATA_PATH = os.path.join("..", "data", "train", "clean_data")
    # VIDEO_EXTENSION = ('.mp4', '.avi')
    # DATA_EXTENSION = ('.npy')

    # read each label video
    for _, label_list, _ in os.walk(TRAIN_RAWDATA_PATH):
        for label in label_list:

            # read each video in each label
            for dirname, _, filenames in os.walk(os.path.join(TRAIN_RAWDATA_PATH, label)):
                for filename in filenames:
                    print(type(filename))
                    exit(1)

if __name__ == '__main__':
    # debug_read_data()
    # y = {'Step 1': '0.00%', 'Step 2': '0.00%', 'Step 3': '1.67%', 'Step 4': '0.00%', 'Step 5': '11.67%', 'Step 6': '78.33%', 'Step 7': '8.33%'}
    # l = []
    # f = lambda k : float(y[k][:-1])
    # for i in y.keys():
    #     l.append(f(i))

    # print(l)
    arr = np.array([[6.0], [6.0], [6.0], [6.0], [6.0], [6.0], [6.0], [6.0], [6.0], [6.0]])
    arr = arr.reshape(arr.shape[0])
    l = arr.tolist()
    print(l)
    print(int(max(set(l), key = l.count)))



