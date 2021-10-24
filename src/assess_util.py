import os, csv
import numpy as np
import matplotlib.pyplot as plt
from main import *
def assess_lost_epoch():
    list_label = {
                  "ltc_1": "LTC",
                  "lstm_1": "LSTM",
                  "ctgru_1": "CT-GRU",
                  "ctrnn_1": "CT-RNN",
                  "node_1": "NODE"
                  }
    for label in list_label.keys():
        path = os.path.join("results", f"{label}",f"10_{label[:-2]}_adhoc.csv")
        cross_loss = []
        train_loss = []
        test_loss = []
        best = [0,0.0]
        cnt = 0
        with open(path,"r") as f:
            csv_reader = csv.reader(f, delimiter=',')
            for row in csv_reader:
                cnt+=1
                if row[0] == "epoch" or row[0].startswith("#"):
                    continue
                
                value = float(row[2]) + 7
                if value > best[1]:
                    best[1] = value
                    best[0] = cnt

                cross_loss.append(float(row[4])+2)
                train_loss.append(float(row[2])+2)
                test_loss.append(float(row[6])+2)

        # plt.plot(range(1,101), cross_loss,  label=f"{list_label[label]}")

        plt.plot(range(1,101), train_loss, color="green", label="train")
        plt.plot(range(1,101), cross_loss, color="blue", label="cross-valid")
        plt.plot(range(1,101), test_loss, color="red", label="test")

    print(best)


    plt.plot([50,50], [0, 100], ':', color='black')
    plt.ylabel("Accuracy (%)")
    # plt.ylabel("Loss") 
    plt.xlabel("Epoch")
    plt.legend(loc="lower right")
    ax = plt.gca()
    # ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00])
    ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100])
    ax.set_xticks([0,10,20,30,40,50,60,70,80,90,100])
    fig = plt.gcf()
    fig.set_size_inches(12.4, 4.8)
    plt.show()
            

def assess_accuracy_each_step():
    path = os.path.join("results", f"ltc_1","forest_size10_real.csv")
    step_cnt = {"Step 1": 0, "Step 2": 0, "Step 3": 0, "Step 4": 0, "Step 5": 0, "Step 6": 0}
    step_acc = {"Step 1": 0, "Step 2": 0, "Step 3": 0, "Step 4": 0, "Step 5": 0, "Step 6": 0}

    with open(path,"r") as f:
        csv_reader = csv.reader(f, delimiter=',')
        for row in csv_reader:
            if row[0] == "target" or row[0].startswith("#"):
                continue
                
            step_cnt[row[0]]+=1
            
            # find most likely predict
            most_likely = [0, 0.0]
            for i in range(1,7):
                cur_step_per = float(str.strip(row[i]))
                if cur_step_per > most_likely[1]:
                    most_likely = [i, cur_step_per]
                
            if row[0][-1] == str(most_likely[0]):
                step_acc[row[0]] += 1
        

    for step in step_acc.keys():
        step_acc[step] = step_acc[step] / step_cnt[step] * 100
    
    plt.bar(step_acc.keys(), step_acc.values(), width=0.8)
    for index,data in enumerate(step_acc.values()):
        plt.text(index-0.35, data + 0.01, f"{round(data,2)}%", color='black', fontdict=dict(fontsize=12))
    plt.ylabel("Percentage")
    ax = plt.gca()
    ax.set_yticks([0,20,40,60,80,100])
    plt.show()
    print(step_acc)

def draw_box_each_step_threshold():
    path = os.path.join("log_output","ltc.log")
    with open(path,"r") as f:
        no_values = [[],[],[],[],[],[],[],[],[],[],[],[]]
        yes_values = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for line in f:
            if line.startswith(">>>"):
                raw_data = [float(e.strip()) for e in line[4:-1].strip().replace('[', '').replace(']', '').split(',')]
                if raw_data[-1] == 0.0:
                    for i in range(12):
                        no_values[i].append(raw_data[i])
                elif raw_data[-1] == 1.0:
                    for i in range(12):
                        yes_values[i].append(raw_data[i])


        plt.boxplot(no_values[6:12])
        plt.ylabel("Score")
        ax = plt.gca()

        ax.set_xticklabels(["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"])


        # axis[1].set_ylim([0,10])
        fig = plt.gcf()
        fig.set_size_inches(6.4, 4.8)
        plt.show()

def average_step_draw():
    path = os.path.join("log_output","ltc.log")
    with open(path,"r") as f:
        no_values = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        yes_values = [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]
        yes_counter = 0
        no_counter = 0
        for line in f:
            if line.startswith(">>>"):
                raw_data = [float(e.strip()) for e in line[4:-1].strip().replace('[', '').replace(']', '').split(',')]
                if raw_data[-1] == 0.0:
                    no_counter += 1
                    for i in range(12):
                        no_values[i] += raw_data[i]
                elif raw_data[-1] == 1.0:
                    yes_counter += 1
                    for i in range(12):
                        yes_values[i] += raw_data[i]

        
        for i in range(12):
            yes_values[i] = yes_values[i]/yes_counter * 100
            no_values[i] = no_values[i]/no_counter * 100

        step_label = ["Step 1", "Step 2", "Step 3", "Step 4", "Step 5", "Step 6"]
        plt.bar(step_label, yes_values[0:6], width=0.8, color="darkorange")    
        for index,data in enumerate(yes_values):
            plt.text(index-0.35, data + 0.02, f"{round(data,2)}%", color='black', fontdict=dict(fontsize=10))
        plt.ylabel("Percentage")
        ax = plt.gca()
        ax.set_yticks([0,10,20,30,40,50,60,70,80,90,100])
        print(yes_values)
        print(no_values)
        plt.show()



def bar_compare_between_network():
    score_labels = ['Yes Precision', 'No Precision', 'Yes Recall', 'No Recall', 'Yes F1-score', 'No F1-score']

    ltc_value = [0.7 ,0.83, 0.88, 0.62 ,0.78, 0.71]
    lstm_value = [0.62 ,0.67, 0.73,0.55 ,0.67, 0.6]
    ctgru_value = [0.67 ,0.71, 0.75,0.62 ,0.71, 0.67]
    ctrnn_value = [0.64 ,0.8, 0.88,0.5 ,0.74, 0.62]
    node_value = [0.57 ,0.56, 0.5,0.62 ,0.53, 0.39]


    x = np.arange(len(score_labels))  # the label locations
    width = 0.15  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 2*width, ltc_value, width, label='LTC', color="blue")
    rects2 = ax.bar(x - width, lstm_value, width, label='LSTM', color="orange") 
    rects3 = ax.bar(x , ctgru_value, width, label='CT-GRU', color="green")  
    rects4 = ax.bar(x + width, ctrnn_value, width, label='CT-RNN', color="red")  
    rects5 = ax.bar(x + 2* width, node_value, width, label='NODE', color="purple")  

    ax.set_xticks(x)
    ax.set_xticklabels(score_labels)
    ax.set_ylabel("Percentage")
    ax.set_xlabel("Metrics by label")
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)

    fig = plt.gcf()
    fig.set_size_inches(12.4, 4.8)
    fig.tight_layout()
    plt.show()

def bar_plot_accuracy():
    network_labels = ['LTC', 'LSTM', 'CT-GRU', 'CT-RNN', 'NODE']
    networks = ['ltc', 'lstm', 'ctgru', 'ctrnn', 'node']
    max_accs = []
    y_values = []
    x_values = range(len(networks))
    for net in networks:
        acc_list = []
        for i in range(100):
            layer2_database = setup_layer2_database()
            layer2_model = SecondLayerModel(LAYER_2_MODEL_TYPE, LAYER2_EPOCH_NUM, LAYER2_ACTIVATION, LAYER2_SOLVER)

            train_model(layer2_model, layer2_database)
            acc = layer2_model.predict(layer2_database)
            acc_list.append(acc)
        
        y_values.append(acc_list)
        max_accs.append(max(set(acc_list), key=acc_list.count))
        print(max(set(acc_list), key=acc_list.count))

    plt.ylabel("Percentage")
    ax = plt.gca()
    ax.set_xticklabels(network_labels)
    plt.boxplot(y_values, positions=x_values)
    plt.show()


assess_lost_epoch()
# assess_accuracy_each_step()
# draw_box_each_step_threshold()
# average_step_draw()
# bar_compare_between_network()
# bar_plot_accuracy()
