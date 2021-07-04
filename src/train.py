import os
import glob
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from tqdm import tqdm

import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU

class DataSet:
    def __init__(self):
        all_x, all_y = self.load_data_from_file()

        # all_x shape: (time step for each layer, number of batch, number of feature)
        self.all_x = np.stack(all_x, axis=1)
        self.all_y = np.stack(all_y, axis=1)
        
        self.divide_dataset()
        print(self.train_x.shape)
        print(self.train_y.shape)
        print(self.test_x.shape)
        print(self.test_y.shape)
        
        
    def load_data_from_file(self):
        DATA_PATH = os.path.join("..", "clean_data")
        DATA_EXTENSION = ('.npy')

        all_x = []
        all_y = []
        
        # read each label in dataset
        print("Reading data...", end="")
        for _, label_list, _ in os.walk(DATA_PATH):
            for label in label_list:
                # print(label)

                # read each video file in each label
                for dirname, _, filenames in os.walk(os.path.join(DATA_PATH, label)):
                    for filename in filenames:
                        if filename.endswith(DATA_EXTENSION):
                            data_path = os.path.join(dirname, filename)
                            # print("\treading: " +data_path)
                            feature = np.load(data_path)
                            all_x.append(feature[:,:-1])
                            all_y.append(feature[:,-1])

        print("Done")                    
        return np.array(all_x), np.array(all_y)

    def iterate_train(self,batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,permutation[start:end]]
            batch_y = self.train_y[:,permutation[start:end]]
            yield (batch_x,batch_y)

    def divide_dataset(self, valid_ratio=0.2, test_ratio=0.1):
        total_seqs = self.all_x.shape[1]
        permutation = np.random.RandomState(27731).permutation(total_seqs)
        valid_size = int(valid_ratio*total_seqs)
        test_size = int(test_ratio*total_seqs)

        self.valid_x = self.all_x[:,permutation[:valid_size]]
        self.valid_y = self.all_y[:,permutation[:valid_size]]
        self.test_x = self.all_x[:,permutation[valid_size:valid_size+test_size]]
        self.test_y = self.all_y[:,permutation[valid_size:valid_size+test_size]]
        self.train_x = self.all_x[:,permutation[valid_size+test_size:]]
        self.train_y = self.all_y[:,permutation[valid_size+test_size:]]
        



class TrainingModel:
    def __init__(self,model_type,model_size,sparsity_level=0.0,learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = []
        self.sparsity_level = sparsity_level
        self.x = tf.placeholder(dtype=tf.float32,shape=[None,None,2772])        ## TODO hyperparameter
        self.target_y = tf.placeholder(dtype=tf.int32,shape=[None,None])

        self.model_size = model_size
        head = self.x
        if(model_type == "lstm"):
            # unstacked_signal = tf.unstack(x,axis=0)
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)

            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            learning_rate = 0.01 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op.extend(self.wm.get_param_constrain_op())
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        if(self.sparsity_level > 0):
            self.constrain_op.extend(self.get_sparsity_ops())

        self.y = tf.layers.Dense(7,activation=None)(head)
        print("logit shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

        self.result_file = os.path.join("results","{}_{}.csv".format(model_type,model_size))
        if(not os.path.exists("results")):
            os.makedirs("results")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        # store the save session
        self.checkpoint_path = os.path.join("tf_sessions",f"{model_type}")
        self.backup_file_name = f"model-{model_type}-size-{model_size}"
        self.load_backup()
        if(not os.path.exists(os.path.join("tf_sessions",f"{model_type}"))):
            os.makedirs(os.path.join("tf_sessions",f"{model_type}"))
            


    def get_sparsity_ops(self):
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
                op_list.append(self.sparse_var(v,self.sparsity_level))
                
        return op_list
        
    def sparse_var(self,v,sparsity_level):
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



    def fit(self,hanwash_data,epochs,verbose=True,log_period=50):
        have_new_best = False
        # self.best_valid_accuracy, self.best_valid_stats= self.load_backup()
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
            for batch_x,batch_y in hanwash_data.iterate_train(batch_size=2):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x:batch_x,self.target_y: batch_y})
                if(len(self.constrain_op) > 0):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if(verbose and e%log_period == 0):
                with open(self.result_file,"a") as f:
                    f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
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

            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break

        with open(os.path.join(self.checkpoint_path, f"{self.backup_file_name}.pickle"), "wb") as bk_f:
            pickle.dump([self.best_valid_accuracy, self.best_valid_stats], bk_f)

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
                f.write("="*30)

    def test(self,hanwash_data):
        # self.best_valid_accuracy, self.best_valid_stats= self.load_backup()

        y_test, test_acc,test_loss = self.sess.run([self.y, self.accuracy,self.loss],{self.x:hanwash_data.test_x,self.target_y: hanwash_data.test_y})
        print("Test result: test loss: {:0.2f}, test accuracy: {:0.2f}%".format(test_loss,test_acc*100))
        print(y_test[0][0].shape)
        print(y_test[0][0])

    def load_backup(self):
        if (os.path.exists(self.checkpoint_path)): 
            #First let's load meta graph and restore weights
            checkpoint_path = os.path.join("tf_sessions",f"{self.model_type}")
            saver = tf.train.import_meta_graph(os.path.join(checkpoint_path, f"model-{self.model_type}-size-{self.model_size}.meta"))
            saver.restore(self.sess,tf.train.latest_checkpoint(checkpoint_path))

            with open(os.path.join(self.checkpoint_path, f"{self.backup_file_name}.pickle"), "rb") as bk_f:
                self.best_valid_accuracy, self.best_valid_stats = pickle.load(bk_f)
        
        else:
            self.best_valid_accuracy = 0
            self.best_valid_stats = (0,0,0,0,0,0,0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',default="train")               # train/test
    parser.add_argument('--model',default="lstm")               # type of Cell for network
    parser.add_argument('--log',default=1,type=int)             # number of iterative for each save
    parser.add_argument('--size',default=10,type=int)           # time step
    parser.add_argument('--epochs',default=200,type=int)        # iterative
    parser.add_argument('--sparsity',default=0.0,type=float)
    args = parser.parse_args()

    dataset = DataSet()
    model = TrainingModel(model_type = args.model,model_size=args.size,sparsity_level=args.sparsity)
    if (args.mode == "train"):
        model.fit(dataset,epochs=args.epochs,verbose=True, log_period=args.log)
    elif (args.mode == "test"):
        model.test(dataset)