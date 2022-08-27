from warnings import simplefilter
from bs4 import SoupStrainer 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report,confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import sys
import pickle
## import config property
from property import *
from layer1_model import *

class VideoSet:
    def __init__(self):        
        self.train_video_x, self.train_video_y = self.load_data_from_file(LAYER_2_TRAIN_PROCESSED_PATH)
        self.valid_video_x, self.valid_video_y = self.load_data_from_file(LAYER_2_VALID_PROCESSED_PATH)
        self.test_video_x, self.test_video_y = self.load_data_from_file(LAYER_2_TEST_PROCESSED_PATH)
        print("Done processing layer2 data")

    def load_data_from_file(self, process_data_path):
        all_x = []
        all_y = []

        # read each label in dataset
        print("Reading data layer1 ...", end="")
        for _, label_list, _ in os.walk(process_data_path):
            for label in label_list:
                for dirname, _, filenames in os.walk(os.path.join(process_data_path, label)):
                    for filename in filenames:
                        if filename.endswith(DATA_EXTENSION):
                            data_path = os.path.join(dirname, filename)
                            video_data = np.load(data_path)
                            all_x.append(video_data[:-1])            
                            all_y.append(video_data[-1])
        
        # Normalize data
        all_x = np.array(all_x)
        score_arr =  all_x[:,6:]
        normalized_score = preprocessing.normalize(score_arr)
        all_x[:,6:] = normalized_score
        return np.array(all_x), np.array(all_y)

               


class SecondLayerModel:
    def __init__(self, model_type, max_iter, activation_type, solver_type):
        self.model_type = model_type
        self.activation_type = activation_type
        self.solver_type = solver_type

        self.result_path = os.path.join("results", "logs", SAVE_LOCATION_NAME, f"{LAYER_2_MODEL_TYPE}_{LAYER2_EPOCH_NUM}_result.txt")
        self.model_save_path = os.path.join("data","trained-model", SAVE_LOCATION_NAME, f"{LAYER_2_MODEL_TYPE}_{LAYER2_EPOCH_NUM}_model_checkpoint.pkl")
        
        if (os.path.exists(self.model_save_path)):
            with open(self.model_save_path, 'rb') as f:
                self.model = pickle.load(f)
        elif (self.model_type == "mlp"):
            self.model = MLPClassifier(hidden_layer_sizes=(12,6,12), activation=self.activation_type, solver=self.solver_type, max_iter=max_iter)
      
    def fit(self, dataset):
        train_x = dataset.train_video_x
        train_y = dataset.train_video_y
        self.model.fit(train_x, train_y)

        # save model
        with open(self.model_save_path,'wb') as f:
            pickle.dump(self.model,f)

    def evaluate(self, data):
        predict_result = self.model.predict(data)
        predict_precent = self.model.predict_proba(data)
        return (predict_result, predict_precent)

    def predict(self, dataset):
        predict_test = self.model.predict(dataset.test_video_x)
        accuracy = 1 - (np.mean( predict_test != dataset.test_video_y ))

        with open(self.result_path,"w") as f:
            f.write(f"accuracy: {accuracy}")
            f.write("\n")
            f.write(str(confusion_matrix(dataset.test_video_y, predict_test)))
            f.write("\n")
            f.write(str(classification_report(dataset.test_video_y, predict_test)))

        print(">>> Accuracy: ", accuracy)

def setup_layer2_model(max_iter):
    return SecondLayerModel(LAYER_2_MODEL_TYPE, max_iter, LAYER2_ACTIVATION, LAYER2_SOLVER)

def setup_layer2_database():
    return VideoSet()

def processing_layer2_feature_data():
    def process_feature_data(training_forest, feature_data_path, process_data_path):
        for _, video_label_list, _ in os.walk(feature_data_path):
            for video_label in video_label_list:
                for dirname, _, filenames in os.walk(os.path.join(feature_data_path, video_label)):
                    # read each video file in each label
                    for filename in filenames:
                        if filename.endswith(DATA_EXTENSION):
                            print(f"processing {video_label}_{os.path.splitext(filename)[0]}.npy")
                            sys.stdout.flush()

                            ## already preprocess:  
                            save_path = os.path.join(process_data_path, video_label,  f"{os.path.splitext(filename)[0]}.npy")
                            if os.path.isfile(save_path):
                                print("skip")
                                continue
                            
                            video_feature = np.load(os.path.join(dirname, filename))
                            _, data = training_forest._process_video(video_feature[:,:-1])
                            feature_vector = [step[0] for step in data] + [step[1] for step in data]
                            
                            feature_vector.append(float(video_feature[0,-1]))    
                            print(">>> ", feature_vector)

                            # save to file
                            np.save(save_path, feature_vector)       
    
    training_forest = setup_layer1_model(MODEL_EPOCH_NUM)

    process_feature_data(training_forest, LAYER_2_TRAIN_PATH, LAYER_2_TRAIN_PROCESSED_PATH)
    process_feature_data(training_forest, LAYER_2_VALID_PATH, LAYER_2_VALID_PROCESSED_PATH)
    process_feature_data(training_forest, LAYER_2_TEST_PATH, LAYER_2_TEST_PROCESSED_PATH)