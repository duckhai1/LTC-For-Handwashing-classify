from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix
import sys
tf.logging.set_verbosity(tf.logging.ERROR)

## import config property
from property import *
from layer1_model import *

class VideoSet:
    def __init__(self, is_regenerate):
        self.training_forest = setup_layer1_model(MODEL_EPOCH_NUM)
        
        self.clean_raw_data(is_regenerate)

        self.all_video_x, self.all_video_y = self.load_data_from_file()
        self._divide_dataset(LAYER_2_VALID_RATIO, LAYER_2_TEST_RATIO)

    def clean_raw_data(self, is_regenerate):
        print("Preparing layer2 clean data...")
        prepare_traindata_destination(LAYER_2_TRAIN_RAWDATA_PATH, LAYER_2_TRAIN_DATA_PATH)  
        prepare_processed_data_destination()
        if (is_regenerate):
            clear_processed_data_destination()
            clear_train_data(LAYER_2_TRAIN_DATA_PATH)
        extract_layer2_train_video()
        print("Done preparing layer2 clean data")

        print("Processing layer2 data...")
        self.process_data_path = os.path.join(LAYER_2_PROCESSED_DATA_PATH, f"{self.training_forest.number_of_tree}_tree_{self.training_forest.model_type}")
        if (not os.path.exists(self.process_data_path)):
            os.makedirs(self.process_data_path)
        self.generate_data()
        print("Done processing layer2 data")

    def load_data_from_file(self):
        all_x = []
        all_y = []
        for dirname, _, filenames in os.walk(self.process_data_path):
            for filename in filenames:
                if filename.endswith(DATA_EXTENSION):
                    data_path = os.path.join(dirname, filename)
                    video_data = np.load(data_path)

                    all_x.append(video_data[:-1])            
                    all_y.append(video_data[-1])
        
        # Normalize data
        all_x = np.array(all_x)
        score_arr =  all_x[:,6::]
        norm = np.linalg.norm(score_arr)
        normal_array = score_arr/norm
        all_x[:,6::] = normal_array
        return np.array(all_x), np.array(all_y)

    def generate_data(self):
        for _, video_label_list, _ in os.walk(LAYER_2_TRAIN_DATA_PATH):
            for video_label in video_label_list:
                for dirname, _, filenames in os.walk(os.path.join(LAYER_2_TRAIN_DATA_PATH, video_label)):
                    # read each video file in each label
                    for filename in filenames:
                        if filename.endswith(DATA_EXTENSION):
                            print(f"processing {video_label}_{os.path.splitext(filename)[0]}.npy")
                            sys.stdout.flush()

                            ## already preprocess:  
                            save_path = os.path.join(self.process_data_path, f"{video_label}_{os.path.splitext(filename)[0]}.npy")
                            if os.path.isfile(save_path):
                                print("skip")
                                continue
                            
                            video_feature = np.load(os.path.join(dirname, filename))
                            _, data = self.training_forest._process_video(video_feature[:,:-1])
                            feature_vector = [step[0] for step in data] + [step[1] for step in data]
                            
                            feature_vector.append(float(video_feature[0,-1]))    
                            print(">>> ", feature_vector)

                            # save to file
                            np.save(save_path, feature_vector)                      


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



class SecondLayerModel:
    def __init__(self, model_type, max_iter, activation_type, solver_type):
        self.model_type = model_type
        self.activation_type = activation_type
        self.solver_type = solver_type

        if (self.model_type == "mlp"):
            self.model = MLPClassifier(hidden_layer_sizes=(12,6,12), activation=self.activation_type, solver=self.solver_type, max_iter=max_iter)

    def fit(self, dataset):
        train_x = dataset.train_video_x
        train_y = dataset.train_video_y
        self.model.fit(train_x, train_y)

    def evaluate(self, data):
        predict_result = self.model.predict(data)
        predict_precent = self.model.predict_proba(data)
        return (predict_result, predict_precent)

    def predict(self, dataset):
        predict_test = self.model.predict(dataset.test_video_x)
        accuracy = 1 - (np.mean( predict_test != dataset.test_video_y ))

        result_path = os.path.join("results", f"{MODEL_TYPE}_{NUMBER_OF_TREE}", f"{LAYER_2_MODEL_TYPE}_{LAYER2_EPOCH_NUM}")
        with open(result_path,"w") as f:
            f.write(f"accuracy: {accuracy}")
            f.write("\n")
            f.write(str(confusion_matrix(dataset.test_video_y, predict_test)))
            f.write("\n")
            f.write(str(classification_report(dataset.test_video_y, predict_test)))

        print(">>> Accuracy: ", accuracy)

def setup_layer2_model(max_iter):
    return SecondLayerModel(LAYER_2_MODEL_TYPE, max_iter, LAYER2_ACTIVATION, LAYER2_SOLVER)

def setup_layer2_database():
    return VideoSet(REGENERATE_LAYER2_DATA)

