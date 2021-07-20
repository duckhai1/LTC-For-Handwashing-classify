from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

tf.logging.set_verbosity(tf.logging.ERROR)

## import config property
from property import *
from layer1_model import *

class VideoSet:
    def __init__(self, is_regenerate):
        self.training_forest = run_layer_1_model_execute_task()
        self.save_data_path = os.path.join(LAYER_2_PROCESSED_DATA_PATH, f"{self.training_forest.number_of_tree}_tree_{self.training_forest.model_type}_output.npy")
        
        if (is_regenerate):
            prepare_traindata_destination(LAYER_2_TRAIN_RAWDATA_PATH, LAYER_2_TRAIN_DATA_PATH)  
            prepare_processed_data_destination()
            clear_train_data(LAYER_2_TRAIN_DATA_PATH)
            extract_layer2_train_video()

        if (not os.path.exists(self.save_data_path)):
            self.generate_data()

        self.all_video_x, self.all_video_y = self.load_data_from_file()
        self._divide_dataset(LAYER_2_VALID_RATIO, LAYER_2_TEST_RATIO)
            
    def load_data_from_file(self):
        video_data = np.load(self.save_data_path)
        all_x = video_data[:,:-1]
        all_y = video_data[:,-1]
        return np.array(all_x), np.array(all_y)

    def generate_data(self):
        all_data = []
        for _, video_label_list, _ in os.walk(LAYER_2_TRAIN_DATA_PATH):
            for video_label in video_label_list:
                for dirname, _, filenames in os.walk(os.path.join(LAYER_2_TRAIN_DATA_PATH, video_label)):
                    # read each video file in each label
                    for filename in filenames:
                        if filename.endswith(DATA_EXTENSION):
                            video_feature = np.load(os.path.join(dirname, filename))
                            data = self._process_x(video_feature[:,:-1])
                            data.append(float(video_feature[0,-1]))                          
                            all_data.append(data)

        # save to file
        np.save(self.save_data_path, all_data)

    def _process_x(self, video_data):
        highest_threshold = {"Step 1": 0., "Step 2": 0., "Step 3": 0., "Step 4": 0., "Step 5": 0., "Step 6": 0., "Step 7": 0.}
        start = 0
        end = start + WINDOWS_LENGTH
        # sliding windows
        while (end <= len(video_data)):
            ## slice the windows
            window_data = video_data[start:end, :]
            ## reshape become data with batch = 1 for feed to model
            window_data = window_data.reshape(window_data.shape[0], 1, window_data.shape[1])
            
            forest_final_predict, forest_percentage = self.training_forest.evaluate(window_data)
            if (highest_threshold[forest_final_predict] < forest_percentage[forest_final_predict]):
                highest_threshold[forest_final_predict] = forest_percentage[forest_final_predict]

            start+=1
            end+=1

            ### DEBUG
            if end == 20:
                break
        
        return list(highest_threshold.values())

    def _divide_dataset(self, valid_ratio, test_ratio):
        total_seqs = self.all_video_x.shape[0]
        permutation = np.random.RandomState(7791).permutation(total_seqs)
        valid_size = int(valid_ratio*total_seqs)
        test_size = int(test_ratio*total_seqs)

        self.valid_video_x = self.all_video_x[permutation[:valid_size]]
        self.valid_video_y = self.all_video_y[permutation[:valid_size]]
        self.test_video_x = self.all_video_x[permutation[valid_size:valid_size+test_size]]
        self.test_video_y = self.all_video_y[permutation[valid_size:valid_size+test_size]]
        self.train_video_x = self.all_video_x[permutation[valid_size+test_size:]]
        self.train_video_y = self.all_video_y[permutation[valid_size+test_size:]]

    def iterate_train(self):
        pass
class SecondLayerModel:
    def __init__(self, model_type):
        if (model_type == "mlp"):
            self.model = MLPClassifier(hidden_layer_sizes=(7,7,7), activation='relu', solver='adam', max_iter=500)

    def fit(self, dataset):
        train_x = dataset.train_video_x
        train_y = dataset.train_video_y
        self.model.fit(train_x, train_y)

    def predict(self, dataset):
        predict_test = self.model.predict(dataset.test_video_x)
        accuracy = 1 - (np.mean( predict_test != dataset.test_video_y ))
        print("accuracy: ", accuracy)
        
        print(confusion_matrix(dataset.test_video_y, predict_test))
        print(classification_report(dataset.test_video_y, predict_test))



if __name__ == '__main__':
    video_set = VideoSet(REGENERATE_LAYER2_DATA)
    outside_model = SecondLayerModel(LAYER_2_MODEL_TYPE)
    outside_model.fit(video_set)
    outside_model.predict(video_set)
