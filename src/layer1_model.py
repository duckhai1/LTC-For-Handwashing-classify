from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)

import os
import glob
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU

## import config property
from property import *
from read_data import *



class DataSet:
    def __init__(self):
        train_x, train_y = self.load_data_from_file(LAYER_1_TRAIN_PATH)
        valid_x, valid_y = self.load_data_from_file(LAYER_1_VALID_PATH)
        test_x, test_y = self.load_data_from_file(LAYER_1_TEST_PATH)

        self.train_x = np.stack(train_x, axis=1)
        self.train_y = np.stack(train_y, axis=1)
        self.valid_x = np.stack(valid_x, axis=1)
        self.valid_y = np.stack(valid_y, axis=1)
        self.test_x = np.stack(test_x, axis=1)
        self.test_y = np.stack(test_y, axis=1)

        self._divide_train_data(NUMBER_OF_TREE)
        print("Done preparing layer1 clean data")
        print("all train_x.shape", self.train_x.shape)
        print("all train_y.shape", self.train_y.shape)
        print("each train_x_set shape", self.train_x_set[0].shape)
        print("each train_y_set shape", self.train_y_set[0].shape)
        print("test_x.shape", self.test_x.shape)
        print("test_y.shape", self.test_y.shape)

        
    def load_data_from_file(self, process_data_path):
        all_x = []
        all_y = []
        
        # read each label in dataset
        print("Reading data layer1 ...", end="")
        for _, label_list, _ in os.walk(process_data_path):
            for label in label_list:
                # read each video file in each label
                for dirname, _, filenames in os.walk(os.path.join(process_data_path, label)):
                    for filename in filenames:
                        if filename.endswith(DATA_EXTENSION):
                            data_path = os.path.join(dirname, filename)
                            feature = np.load(data_path)
                            all_x.append(feature[:,:-1])
                            all_y.append(feature[:,-1])

        print("Done")                    
        return np.array(all_x), np.array(all_y)

    def iterate_train(self,train_set_number, batch_size):
        train_x = self.train_x_set[train_set_number]
        train_y = self.train_y_set[train_set_number]
        total_seqs = train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = train_x[:,permutation[start:end]]
            batch_y = train_y[:,permutation[start:end]]
            yield (batch_x,batch_y)

    def _divide_train_data(self, number_of_set):
        self.train_x_set = []
        self.train_y_set = []
        total_seqs = self.train_x.shape[1]
        set_size = total_seqs // number_of_set
        permutation = np.random.RandomState(number_of_set).permutation(total_seqs)
        for s in range(number_of_set):
            start = s*set_size
            end = start + set_size
            set_x = self.train_x[:,permutation[start:end]]
            set_y = self.train_y[:,permutation[start:end]]
            self.train_x_set.append(set_x)
            self.train_y_set.append(set_y)

    def iterate_test(self):
        total_seqs = self.test_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        for i in range(total_seqs):
            yield (self.test_x[:,permutation[[i]]], self.test_y[:,permutation[[i]]])


class TrainingModel:
    def __init__(self,train_set_number, model_type,model_size,sparsity_level):
        self.train_set_number = train_set_number
        self.model_type = model_type
        self.sparsity_level = sparsity_level
        self.model_size = model_size
        self.learning_rate = 0.001

        self.constrain_op = []
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,2772])
        self.target_y = tf.placeholder(dtype=tf.int32,shape=[None,None])
        head = self.x

        if(self.model_type == "lstm"):
            # unstacked_signal = tf.unstack(x,axis=0)
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(self.model_size)

            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(self.model_type.startswith("ltc")):
            self.learning_rate = 0.01 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(self.model_size)
            if(self.model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(self.model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op.extend(self.wm.get_param_constrain_op())
        elif(self.model_type == "node"):
            self.fused_cell = NODE(self.model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(self.model_type == "ctgru"):
            self.fused_cell = CTGRU(self.model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(self.model_type == "ctrnn"):
            self.fused_cell = CTRNN(self.model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(self.model_type))

        if(self.sparsity_level > 0):
            self.constrain_op.extend(self._get_sparsity_ops())

        self.y = tf.layers.Dense(6,activation=None)(head)
        # print("logit y shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        self.predict_percent = tf.argsort(self.y, axis=2)

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.result_path = os.path.join("results", "logs", SAVE_LOCATION_NAME)
        self.result_file = os.path.join("results", "logs", SAVE_LOCATION_NAME,f"{self.model_size}_{self.model_type}_{train_set_number}.csv")
        if(not os.path.exists(self.result_path)):
            os.makedirs(self.result_path)
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        # store the save session
        self.checkpoint_path = os.path.join("data", "trained-model",SAVE_LOCATION_NAME, f"model-{train_set_number}")
        self.backup_file_name = f"{train_set_number}-{self.model_type}-size-{self.model_size}"
        self.load_backup()
        if(not os.path.exists(self.checkpoint_path)):
            os.makedirs(self.checkpoint_path)
            

    def _get_sparsity_ops(self):
        tf_vars = tf.trainable_variables()
        op_list = []
        for v in tf_vars:
            # print("Variable {}".format(str(v)))
            if(v.name.startswith("rnn")):
                if(len(v.shape)<2):
                    # Don't sparsity biases
                    continue
                if("ltc" in v.name and (not "W:0" in v.name)):
                    # LTC can be sparsified by only setting w[i,j] to 0
                    # both input and recurrent matrix will be sparsified
                    continue
                op_list.append(self._sparse_var(v,self.sparsity_level))
                
        return op_list
        
    def _sparse_var(self,v,sparsity_level):
        mask = np.random.choice([0, 1], size=v.shape, p=[sparsity_level,1-sparsity_level]).astype(np.float32)
        v_assign_op = tf.assign(v,v*mask)
        print("Var[{}] will be sparsified with {:0.2f} sparsity level".format(
            v.name,sparsity_level
        ))
        return v_assign_op

    def save(self):
        self.saver.save(self.sess, os.path.join(self.checkpoint_path, self.backup_file_name))

    def restore(self):
        self.saver.restore(self.sess, os.path.join(self.checkpoint_path, self.backup_file_name))

    def load_backup(self):
        if (os.path.exists(os.path.join(self.checkpoint_path, f"{self.backup_file_name}.meta"))): 
            #First let's load meta graph and restore weights
            saver = tf.train.import_meta_graph(os.path.join(self.checkpoint_path, f"{self.backup_file_name}.meta"))
            saver.restore(self.sess,tf.train.latest_checkpoint(self.checkpoint_path))

            with open(os.path.join(self.checkpoint_path, f"{self.backup_file_name}.pickle"), "rb") as bk_f:
                self.best_valid_accuracy, self.best_valid_stats = pickle.load(bk_f)
        
        else:
            self.best_valid_accuracy = 0
            self.best_valid_stats = (0,0,0,0,0,0,0)


    def fit(self,hanwash_data,epochs,verbose=True,log_period=50):
        have_new_best = False
        self.save()
        print("Entering training loop")
        for e in range(epochs):
            # log the duration training result
            if(e%log_period == 0):
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x:hanwash_data.test_x,self.target_y: hanwash_data.test_y})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:hanwash_data.valid_x,self.target_y: hanwash_data.valid_y})
                if(valid_acc > self.best_valid_accuracy and e > 0):
                    have_new_best = True
                    self.best_valid_accuracy = valid_acc
                    self.best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs)*100,
                        valid_loss,valid_acc*100,
                        test_loss,test_acc*100
                    )
                    self.save()
            
            # training
            losses = []
            accs = []
            for batch_x,batch_y in hanwash_data.iterate_train(self.train_set_number, BATCH_SIZE):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x:batch_x,self.target_y: batch_y})
                if(len(self.constrain_op) > 0):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if(verbose and e%log_period == 0):
                with open(self.result_file,"a") as f:
                    f.write("{:03d},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f},{:0.2f}\n".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
                print("Epochs {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))

                with open(os.path.join(self.checkpoint_path, f"{self.backup_file_name}.pickle"), "wb") as bk_f:
                    pickle.dump([self.best_valid_accuracy, self.best_valid_stats], bk_f)

            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break


        if (have_new_best):
            best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = self.best_valid_stats
            print("# Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                best_epoch,
                train_loss,train_acc,
                valid_loss,valid_acc,
                test_loss,test_acc
            ))
            with open(self.result_file,"a") as f:
                f.write("# Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%\n".format(
                best_epoch,
                train_loss,train_acc,
                valid_loss,valid_acc,
                test_loss,test_acc
                ))
                f.write("#" + "="*110 + "\n")

    def evaluate(self, data):
        result_dict = {"Step 1": [0, 0.], "Step 2": [0, 0.], "Step 3": [0, 0.], "Step 4": [0, 0.], "Step 5": [0, 0.], "Step 6": [0, 0.], "Total": 0}
        order_lists, score_lists = self.sess.run([self.predict_percent, self.y], {self.x:data})
        order_lists = order_lists.reshape([10,6])
        score_lists = score_lists.reshape([10,6])

        for o_list, s_list in zip(order_lists, score_lists):
            result_dict["Total"] += 1 
            target = o_list[-1]
            score = s_list[target]

            if target == 0:
                result_dict["Step 1"][0] += 1 
                result_dict["Step 1"][1] += score
            elif target == 1:
                result_dict["Step 2"][0] += 1 
                result_dict["Step 2"][1] += score
            elif target == 2:
                result_dict["Step 3"][0] += 1 
                result_dict["Step 3"][1] += score
            elif target == 3:
                result_dict["Step 4"][0] += 1 
                result_dict["Step 4"][1] += score
            elif target == 4:
                result_dict["Step 5"][0] += 1 
                result_dict["Step 5"][1] += score
            elif target == 5:
                result_dict["Step 6"][0] += 1
                result_dict["Step 6"][1] += score

        return result_dict


class TrainingForest:   
    def __init__(self, number_of_tree, tree_list, model_size, sparsity_level, epochs, log_period):
        self.number_of_tree = number_of_tree
        self.tree_list = tree_list
        self.sparsity_level = sparsity_level
        self.model_size = model_size
        self.epochs = epochs
        self.log_period = log_period

        # save result to file
        self.result_path = os.path.join("results", "logs", SAVE_LOCATION_NAME)
        self.result_file = os.path.join("results", "logs", SAVE_LOCATION_NAME,"forest_size{}.csv".format(self.model_size))
        if(not os.path.exists(self.result_path)):
            os.makedirs(self.result_path)
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("target, step_1, step_2, step_3, step_4, step_5, step_6, score_1, score_2, score_3, score_4, score_5, score_6\n")

    def _map_test_result(self, result_list):
        # convert to list of result 
        result_list = result_list.reshape(result_list.shape[0]).tolist()
        # get the max frequent result
        result = int(max(set(result_list), key = result_list.count))
        result_map = {0: "Step 1", 1: "Step 2", 2: "Step 3", 3: "Step 4", 4: "Step 5", 5: "Step 6"}
        return result_map[result]

    def _process_video(self, video_data):
        highest_threshold = {"Step 1": [0., 0.], "Step 2": [0., 0.], "Step 3": [0., 0.], "Step 4": [0., 0.], "Step 5": [0., 0.], "Step 6": [0., 0.]}
        start = 0
        end = start + WINDOWS_LENGTH
        # sliding windows
        while (end <= len(video_data)):
            ## slice the windows
            window_data = video_data[start:end, :]
            ## reshape become data with batch = 1 for feed to model
            window_data = window_data.reshape(window_data.shape[0], 1, window_data.shape[1])
            
            forest_final_predict, forest_percentage = self.evaluate(window_data)
            print("\t", forest_percentage)
            
            for step in forest_percentage.keys():
                # sum score
                highest_threshold[step][1] += forest_percentage[step][1]
                # max occurrence 
                if (highest_threshold[step][0] < forest_percentage[step][0]):
                    highest_threshold[step][0] = forest_percentage[step][0]

            start+=1
            end+=1
        
        # average occurrence
        for step in highest_threshold.keys():
            highest_threshold[step][1] = highest_threshold[step][1] / start

        return (forest_final_predict, list(highest_threshold.values()))

    def fit(self, dataset):
        for model_num in range(self.number_of_tree):
            model = TrainingModel(train_set_number=model_num, model_type = self.tree_list[model_num], model_size=self.model_size, sparsity_level=self.sparsity_level)
            model.fit(dataset, epochs=self.epochs,verbose=True, log_period=self.log_period)

            model.sess.close()
            tf.reset_default_graph()

        print("="*100)
        print("Finish training model, the last result can be found in 'result' folder")

    def evaluate(self, data):
        forest_result = {"Step 1": [0, 0.], "Step 2": [0, 0.], "Step 3": [0, 0.], "Step 4": [0, 0.], "Step 5": [0, 0.], "Step 6": [0, 0.], "Total": 0}

        for model_num in range(self.number_of_tree):
            model = TrainingModel(train_set_number=model_num, model_type = self.tree_list[model_num], model_size=self.model_size, sparsity_level=self.sparsity_level)
            tree_result = model.evaluate(data)

            # Adding result
            for step in forest_result.keys():
                if step == "Total":
                    forest_result[step] += tree_result[step]
                else:
                    forest_result[step][0] += tree_result[step][0]
                    forest_result[step][1] += tree_result[step][1]

            model.sess.close()
            tf.reset_default_graph()

        forest_percent = {}
        for step in forest_result.keys():
            if step == "Total":
                continue
            step_pct = forest_result[step][0] / forest_result["Total"]
            if forest_result[step][0] == 0:
                step_score = 0
            else:
                step_score = forest_result[step][1] / forest_result[step][0]
            forest_percent[step] = [round(step_pct, 4), step_score]
 
        forest_final_predict = max(forest_percent, key=lambda k: forest_percent[k][0])
        return (forest_final_predict, forest_percent)

    def predict(self, dataset):
        total_test_data = 0
        correct_predict = 0
        for test_x,test_y in dataset.iterate_test():
            total_test_data += 1
            test_y = self._map_test_result(test_y)
            forest_final_predict, forest_percent = self.evaluate(test_x)

            # save to file
            with open(self.result_file,"a") as f:
                f.write("{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                test_y,
                forest_percent["Step 1"][0],
                forest_percent["Step 2"][0],
                forest_percent["Step 3"][0],
                forest_percent["Step 4"][0],
                forest_percent["Step 5"][0],
                forest_percent["Step 6"][0],
                forest_percent["Step 1"][1],
                forest_percent["Step 2"][1],
                forest_percent["Step 3"][1],
                forest_percent["Step 4"][1],
                forest_percent["Step 5"][1],
                forest_percent["Step 6"][1]
                ))

            if (forest_final_predict == test_y):
                correct_predict += 1
        
        accuracy = correct_predict/total_test_data * 100
        with open(self.result_file,"a") as f:
            f.write(f"# Training forest accuracy: {accuracy}%")
            f.write("#" + "="*100)
            
        print(f"Training forest accuracy: {accuracy}%")


def setup_layer1_model(max_iter):
    return TrainingForest(NUMBER_OF_TREE, TREE_TYPE_LIST, MODEL_SIZE, MODEL_SPARSITY, max_iter, MODEL_LOG_PERIOD)

def setup_layer1_database():
    return DataSet()

