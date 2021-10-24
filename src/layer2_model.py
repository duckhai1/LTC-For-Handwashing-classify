from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pandas as pd 
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix, plot_roc_curve, plot_precision_recall_curve
import sys
import pickle
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
        self.process_data_path = os.path.join(LAYER_2_PROCESSED_DATA_PATH, SAVE_LOCATION_NAME)
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
        score_arr =  all_x[:,6:]
        normalized_score = preprocessing.normalize(score_arr)
        all_x[:,6:] = normalized_score
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
        permutation = np.random.permutation(total_seqs)
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

        self.result_path = os.path.join("results", SAVE_LOCATION_NAME, f"{LAYER_2_MODEL_TYPE}_{LAYER2_EPOCH_NUM}_result.txt")
        self.model_save_path = os.path.join("tf_sessions", SAVE_LOCATION_NAME, f"{LAYER_2_MODEL_TYPE}_{LAYER2_EPOCH_NUM}_model_checkpoint.pkl")
        
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
        accuracy = metrics.accuracy_score(dataset.test_video_y, predict_test)        

        with open(self.result_path,"w") as f:
            f.write(f"accuracy: {accuracy}")
            f.write("\n")
            f.write(str(confusion_matrix(dataset.test_video_y, predict_test)))
            f.write("\n")
            f.write(str(classification_report(dataset.test_video_y, predict_test)))

        self.draw_roc_graph(dataset)
        print(">>> Accuracy: ", accuracy)

    def draw_roc_graph(self, dataset):
        y_pred_proba = self.model.predict_proba(dataset.test_video_x)[::,1]
        fpr, tpr, threshold = metrics.roc_curve(dataset.test_video_y, y_pred_proba)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        
        plt.legend(loc = 'lower right')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    def draw_pr_graph(self, dataset):
        predict_test = self.model.predict(dataset.test_video_x)

        y_pred_proba_rc = self.model.predict_proba(dataset.test_video_x)[::,1]
        prec, recall, threshold = metrics.precision_recall_curve(dataset.test_video_y , y_pred_proba_rc)
        ap_auc = metrics.average_precision_score(dataset.test_video_y , y_pred_proba_rc)
        plt.step(recall, prec, 'b', label = 'AP = %0.2f' % ap_auc)
        plt.plot([0, 1], [1, 0],'r--')
        plt.legend(loc = 'lower left')
        plt.ylabel('Precision')
        plt.xlabel('Recall')
        plt.show()


def setup_layer2_model(max_iter):
    return SecondLayerModel(LAYER_2_MODEL_TYPE, max_iter, LAYER2_ACTIVATION, LAYER2_SOLVER)

def setup_layer2_database():
    return VideoSet(REGENERATE_LAYER2_DATA)

