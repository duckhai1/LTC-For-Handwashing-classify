from layer1_model import *
from layer2_model import *
from generating_raw_data import *
import argparse


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
    parser.add_argument('-l','--layer', help="Choose which layer to run on: layer1/layer2", default="setup", required=True)
    parser.add_argument('--train', help="Set train mode", default="true")
    parser.add_argument('--test', help="Set test mode", default="true")
    parser.add_argument('--eval', help="Set eval mode",default="false")
    parser.add_argument('-p', '--path',  help="Set video path for evaluation")
    args = parser.parse_args()



    if args.layer == "layer1":
        layer1_database = setup_layer1_database()
        layer1_model = setup_layer1_model(MODEL_EPOCH_NUM)

        if args.train == "true":
            train_model(layer1_model, layer1_database)
        if args.test == "true":
            test_model(layer1_model, layer1_database)
        if args.eval == "true":
            eval_layer1(layer1_model, args.path)

    elif args.layer == "layer2":
        layer2_database = setup_layer2_database()
        layer2_model = setup_layer2_model(LAYER2_EPOCH_NUM)

        if args.train == "true":
            train_model(layer2_model, layer2_database)
        if args.test == "true":
            test_model(layer2_model, layer2_database)
        if args.eval == "true":
            layer1_model = setup_layer1_model(MODEL_EPOCH_NUM)
            eval_layer2(layer1_model, layer2_model, args.path)
    else:
        print("Wrong usage")
