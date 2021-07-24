from layer1_model import *
from layer2_model import *
from generating_raw_data import *


def train_model(model, database):
    model.fit(database)

def test_model(model, database):
    model.predict(database)


def eval_layer1(model, vid_path):
    frame_list = extract_single_vid(vid_path)
    predict_step, data = model._process_video(model, frame_list)
    return (predict_step, data)

def eval_layer2(layer1_model, layer2_model, vid_path):
    predict_step, data = eval_layer1(layer1_model, vid_path)
    predict_result, predict_precent = layer2_model.evaluate(data)
    return (predict_result, predict_precent)

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--mode',default="setup")
    parser.add_argument('-p','--path')
    args = parser.parse_args()


    layer1_database = setup_layer1_database()
    # layer2_database = setup_layer2_database()
    
    layer1_model = setup_layer1_model(MODEL_EPOCH_NUM)
    # layer2_model = setup_layer2_model(LAYER2_EPOCH_NUM)

    if args.mode == "train_1":
        # train layer_1
        train_model(layer1_model, layer1_database)
    elif args.mode == "train_2":
        # train layer_2
        train_model(layer2_model, layer2_database, 1, 0)
    elif args.mode == "test_1":
        # test layer_1
        test_model(layer1_model, layer1_database)
    elif args.mode == "test_2":
        test_model(layer2_model, layer2_database)
    elif args.mode == "eval_1":
        eval_layer1(layer1_model, args.path)
    elif args.mode == "eval_2":
        eval_layer2(layer1_model, layer2_model, args.path)
    elif args.mode == "setup":
        print("Finish setup database")
    else:
        print("Wrong usage")
